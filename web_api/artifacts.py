"""Contained, quota-bound registry for job output artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import secrets
import zipfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import BinaryIO
from urllib.parse import quote

from .contracts import ArtifactRef


class ArtifactError(ValueError):
    """Raised when an artifact cannot be safely registered or served."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when an artifact id is not present in the registry."""


@dataclass(frozen=True)
class RegisteredArtifact:
    ref: ArtifactRef
    path: Path


def _has_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


class ArtifactRegistry:
    """Track only server-created regular files beneath one job directory."""

    def __init__(
        self,
        root: Path,
        *,
        max_files: int = 16,
        max_file_bytes: int = 16 * 1024 * 1024,
        max_total_bytes: int = 48 * 1024 * 1024,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self._artifacts: dict[str, RegisteredArtifact] = {}

    @property
    def artifacts(self) -> list[ArtifactRef]:
        return [item.ref for item in sorted(self._artifacts.values(), key=lambda item: item.ref.name)]

    def _validate_name(self, name: str) -> str:
        if (
            not name
            or _has_control_characters(name)
            or "/" in name
            or "\\" in name
            or name == "manifest.json"
        ):
            raise ArtifactError("artifact filename is unsafe")
        return name

    def _validate_media_type(self, media_type: str | None) -> str | None:
        if media_type is None:
            return None
        if not media_type.strip() or _has_control_characters(media_type):
            raise ArtifactError("artifact media type is unsafe")
        return media_type

    def _resolve_candidate(self, relative_path: str | os.PathLike[str]) -> Path:
        raw = os.fspath(relative_path)
        if not isinstance(raw, str) or not raw:
            raise ArtifactError("artifact path is invalid")
        if _has_control_characters(raw) or PureWindowsPath(raw).is_absolute():
            raise ArtifactError("artifact path is unsafe")
        candidate_input = Path(raw)
        if candidate_input.is_absolute() or ".." in candidate_input.parts:
            raise ArtifactError("artifact path is unsafe")
        candidate = self.root / candidate_input
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ArtifactError("artifact path is unavailable") from exc
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactError("artifact path escapes job directory") from exc
        current = candidate
        while True:
            if current.is_symlink():
                raise ArtifactError("symlink artifacts are not allowed")
            if current == self.root:
                break
            current = current.parent
        if not resolved.is_file() or resolved.is_dir():
            raise ArtifactError("artifact must be a regular file")
        return resolved

    def _stream_metadata(self, path: Path) -> tuple[int, str]:
        try:
            initial_size = path.stat().st_size
        except OSError as exc:
            raise ArtifactError("artifact cannot be read") from exc
        if initial_size > self.max_file_bytes:
            raise ArtifactError("artifact exceeds per-file quota")
        digest = hashlib.sha256()
        total = 0
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_file_bytes:
                        raise ArtifactError("artifact exceeds per-file quota")
                    digest.update(chunk)
            if path.stat().st_size != total or total != initial_size:
                raise ArtifactError("artifact changed while it was registered")
        except OSError as exc:
            raise ArtifactError("artifact cannot be read") from exc
        return total, digest.hexdigest()

    def register(
        self,
        relative_path: str | os.PathLike[str],
        *,
        original_name: str | None = None,
        media_type: str | None = None,
    ) -> ArtifactRef:
        if len(self._artifacts) >= self.max_files:
            raise ArtifactError("artifact count quota exceeded")
        path = self._resolve_candidate(relative_path)
        if any(item.path == path for item in self._artifacts.values()):
            raise ArtifactError("artifact is already registered")
        name = self._validate_name(original_name if original_name is not None else path.name)
        if any(item.ref.name == name for item in self._artifacts.values()):
            raise ArtifactError("artifact filename is already registered")
        size_bytes, sha256 = self._stream_metadata(path)
        total_bytes = sum(item.ref.size_bytes for item in self._artifacts.values())
        if total_bytes + size_bytes > self.max_total_bytes:
            raise ArtifactError("artifact total quota exceeded")
        artifact_id = secrets.token_urlsafe(12)
        while artifact_id in self._artifacts:
            artifact_id = secrets.token_urlsafe(12)
        validated_media_type = self._validate_media_type(media_type)
        inferred_media_type = validated_media_type or mimetypes.guess_type(name, strict=False)[0] or "application/octet-stream"
        ref = ArtifactRef(
            id=artifact_id,
            name=name,
            media_type=inferred_media_type,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        self._artifacts[artifact_id] = RegisteredArtifact(ref=ref, path=path)
        return ref

    def get(self, artifact_id: str) -> RegisteredArtifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise ArtifactNotFoundError("artifact not found") from exc

    def resolve_for_download(self, artifact_id: str) -> RegisteredArtifact:
        item = self.get(artifact_id)
        path = self._resolve_candidate(item.path.relative_to(self.root))
        if path != item.path:
            raise ArtifactError("artifact path changed")
        try:
            stat = path.stat()
        except OSError as exc:
            raise ArtifactError("artifact is unavailable") from exc
        if stat.st_size != item.ref.size_bytes:
            raise ArtifactError("artifact changed after registration")
        _size, sha256 = self._stream_metadata(path)
        if sha256 != item.ref.sha256:
            raise ArtifactError("artifact changed after registration")
        return item

    def build_zip(self) -> BinaryIO:
        output = io.BytesIO()
        entries = [self.resolve_for_download(item.ref.id) for item in self._artifacts.values()]
        entries.sort(key=lambda item: item.ref.name)
        manifest = {
            "artifacts": [item.ref.model_dump(mode="json") for item in entries],
        }
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            manifest_info = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            manifest_info.create_system = 3
            manifest_info.external_attr = 0o644 << 16
            archive.writestr(
                manifest_info,
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            )
            for item in entries:
                info = zipfile.ZipInfo(item.ref.name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o644 << 16
                with item.path.open("rb") as handle:
                    archive.writestr(info, handle.read())
        output.seek(0)
        return output

    def content_disposition(self, artifact_id: str) -> str:
        item = self.resolve_for_download(artifact_id)
        return f"attachment; filename*=UTF-8''{quote(item.ref.name, safe='')}"


__all__ = ["ArtifactError", "ArtifactNotFoundError", "ArtifactRegistry", "RegisteredArtifact"]

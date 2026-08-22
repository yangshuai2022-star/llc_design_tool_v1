from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from web_api.artifacts import ArtifactError, ArtifactRegistry


def test_registry_rejects_unsafe_or_duplicate_paths(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    source = root / "report.txt"
    source.write_text("report", encoding="utf-8")
    (root / "other.txt").write_text("other", encoding="utf-8")
    (root / "third.txt").write_text("third", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (root / "folder").mkdir()
    symlink = root / "link.txt"
    symlink.symlink_to(outside)
    registry = ArtifactRegistry(root)

    artifact = registry.register("report.txt")
    assert artifact.name == "report.txt"
    with pytest.raises(ArtifactError):
        registry.register("report.txt")
    with pytest.raises(ArtifactError):
        registry.register("../outside.txt")
    with pytest.raises(ArtifactError):
        registry.register(outside)
    with pytest.raises(ArtifactError):
        registry.register("folder")
    with pytest.raises(ArtifactError):
        registry.register("link.txt")
    with pytest.raises(ArtifactError):
        registry.register("bad\nname.txt")
    with pytest.raises(ArtifactError):
        registry.register("other.txt", original_name="manifest.json")
    with pytest.raises(ArtifactError):
        registry.register("third.txt", media_type="text/plain\r\nX-Leak: yes")
    with pytest.raises(ArtifactError):
        registry.register("other.txt", original_name=".")
    with pytest.raises(ArtifactError):
        registry.register("other.txt", original_name="..")


def test_registry_preserves_name_hashes_and_builds_deterministic_zip(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    (root / "z.txt").write_text("z", encoding="utf-8")
    (root / "a.csv").write_text("a,1\n", encoding="utf-8")
    registry = ArtifactRegistry(root)
    registry.register("z.txt", original_name="server-generated-z.txt")
    registry.register("a.csv")

    first = registry.build_zip()
    second = registry.build_zip()
    assert first.getvalue() == second.getvalue()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["manifest.json", "a.csv", "server-generated-z.txt"]
        manifest = json.loads(archive.read("manifest.json"))
        assert [item["name"] for item in manifest["artifacts"]] == ["a.csv", "server-generated-z.txt"]
        assert "manifest.json" not in [item["name"] for item in manifest["artifacts"]]


def test_registry_rechecks_containment_before_download(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    source = root / "report.txt"
    source.write_text("report", encoding="utf-8")
    registry = ArtifactRegistry(root)
    artifact = registry.register("report.txt")
    source.unlink()
    source.symlink_to(tmp_path / "outside.txt")
    with pytest.raises(ArtifactError):
        registry.resolve_for_download(artifact.id)


def test_registry_rechecks_digest_and_rejects_duplicate_names(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    first = root / "first.txt"
    second = root / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    registry = ArtifactRegistry(root)
    registry.register("first.txt", original_name="report.txt")
    with pytest.raises(ArtifactError):
        registry.register("second.txt", original_name="report.txt")
    first.write_text("two", encoding="utf-8")
    artifact = registry.artifacts[0]
    with pytest.raises(ArtifactError):
        registry.resolve_for_download(artifact.id)


def test_registry_enforces_file_and_total_quotas(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    registry = ArtifactRegistry(root, max_file_bytes=4, max_total_bytes=5, max_files=2)
    (root / "one.txt").write_bytes(b"1234")
    (root / "two.txt").write_bytes(b"12")
    registry.register("one.txt")
    with pytest.raises(ArtifactError):
        registry.register("two.txt")
    (root / "large.txt").write_bytes(b"12345")
    with pytest.raises(ArtifactError):
        registry.register("large.txt")

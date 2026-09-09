import json

from llc_design import __version__
from llc_design.core.config import load_spec, save_project, save_spec
from llc_design.core.spec import LLCDesignSpec, PrimaryTopology
from llc_design.models.system import LLCSystemAnalyzer


def test_config_round_trip(tmp_path):
    source = LLCDesignSpec(primary_topology=PrimaryTopology.HALF_BRIDGE,
                           primary_turns=16)
    path = save_spec(source, tmp_path / "spec.json")
    loaded = load_spec(path)
    assert loaded.primary_topology == PrimaryTopology.HALF_BRIDGE
    assert loaded.primary_turns == 16
    assert loaded.vout_v == 53.0


def test_project_document_is_versioned_and_backwards_loadable(tmp_path):
    source = LLCDesignSpec(primary_turns=28)
    path = save_project(source, tmp_path / "project.json")
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema"] == "power-design-toolkit/project"
    assert document["schema_version"] == 1
    assert document["toolkit_version"] == __version__
    assert document["analysis"]["status"] == "not_run"
    assert document["data_provenance"]["hardware_release_ready"] is False
    assert load_spec(path).primary_turns == 28


def test_project_document_serializes_completed_analysis(tmp_path):
    source = LLCDesignSpec()
    analysis = LLCSystemAnalyzer().analyze(source)
    path = save_project(source, tmp_path / "project.json", analysis)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["analysis"]["status"] == "complete"
    assert document["analysis"]["feasible"] is True
    assert document["analysis"]["nominal"]["efficiency"] > 0.9

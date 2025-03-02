import json
from pathlib import Path

import pytest

from memorygate.cli import main

ROOT = Path(__file__).resolve().parents[1]


def small_baseline(tmp_path, adapter="better"):
    path = tmp_path / "base.json"
    assert main(["baseline", "--adapter", adapter, "--customers", "25", "--out", str(path)]) == 0
    return path


def test_gate_passes_for_same_adapter_and_writes_reports(tmp_path, capsys):
    base = small_baseline(tmp_path)
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    code = main(["gate", "--adapter", "better", "--baseline", str(base),
                 "--report-md", str(md), "--report-json", str(js)])
    assert code == 0
    assert "PASSED" in md.read_text()
    assert json.loads(js.read_text())["passed"] is True


def test_gate_fails_for_worse_adapter(tmp_path):
    base = small_baseline(tmp_path)
    assert main(["gate", "--adapter", "worse", "--baseline", str(base)]) == 1


def test_gate_errors_without_baseline_or_with_bad_adapter(tmp_path, capsys):
    assert main(["gate", "--adapter", "better", "--baseline", str(tmp_path / "missing.json")]) == 2
    base = small_baseline(tmp_path)
    assert main(["gate", "--adapter", "nope", "--baseline", str(base)]) == 2
    assert "error" in capsys.readouterr().err


def test_generator_change_is_detected(tmp_path):
    base = small_baseline(tmp_path)
    data = json.loads(base.read_text())
    data["dataset_fingerprint"] = "stale"
    base.write_text(json.dumps(data))
    assert main(["gate", "--adapter", "better", "--baseline", str(base)]) == 2


def test_generate_and_adapters_commands(tmp_path, capsys):
    out = tmp_path / "ds.json"
    assert main(["generate", "--customers", "5", "--out", str(out)]) == 0
    assert "conversations" in json.loads(out.read_text())
    assert main(["adapters"]) == 0
    assert "ollama" in capsys.readouterr().out


def test_committed_baseline_gates_demo_adapters():
    """Guards the shipped baseline (and generator/adapter drift)."""
    baseline = str(ROOT / "baselines" / "baseline.json")
    assert main(["gate", "--adapter", "better", "--baseline", baseline]) == 0
    assert main(["gate", "--adapter", "worse", "--baseline", baseline]) == 1

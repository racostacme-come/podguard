import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from podguard.cli import main
from podguard.study import evaluation_parameters, run_study, training_parameters


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    output = tmp_path_factory.mktemp("audit")
    summary = run_study(output, n=15, rank=16)
    return output, summary


def test_study_science_and_serialized_artifacts(audit):
    output, summary = audit
    assert all(summary["checks"].values())
    assert summary["max_held_out_relative_l2_error"] < 0.001
    assert summary["max_shifted_relative_l2_error"] > summary["max_held_out_relative_l2_error"]
    assert summary["min_l2_bound_to_error"] >= 1
    assert json.loads((output / "validation.json").read_text()) == summary
    assert (output / "podguard_audit.png").read_bytes().startswith(b"\x89PNG")
    with (output / "audit.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48 * 5
    field = np.genfromtxt(output / "field.csv", delimiter=",", names=True)
    assert field.size == 225
    assert np.isfinite(field["reduced"]).all()
    assert summary["median_online_seconds"] > 0


def test_no_training_evaluation_leakage_and_fixed_seed():
    training = set(training_parameters())
    evaluation = evaluation_parameters()
    assert evaluation == evaluation_parameters()
    assert all(p not in training for _, p in evaluation)
    assert len(set(p for _, p in evaluation)) == 48


def test_cli_invalid_input(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--n", "1", "--output", str(tmp_path)])
    assert exc.value.code == 2
    assert "n must be" in capsys.readouterr().err


def test_cli_end_to_end(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "podguard.cli",
            "--n",
            "9",
            "--rank",
            "8",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert all(json.loads(result.stdout)["checks"].values())


def test_output_path_error(tmp_path):
    path = tmp_path / "file"
    path.write_text("existing file")
    with pytest.raises(SystemExit) as exc:
        main(["--n", "5", "--rank", "2", "--output", str(path)])
    assert exc.value.code == 2

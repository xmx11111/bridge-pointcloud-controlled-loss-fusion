import json
from pathlib import Path

from bridge_seg.cli import main


def test_smoke_cli_writes_artifact(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"

    assert main(["smoke", "--output", str(output), "--seed", "7"]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["artifact_kind"] == "synthetic_smoke"
    assert payload["placeholder_labels"] is True

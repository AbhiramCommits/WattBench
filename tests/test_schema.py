"""Results schema round-trip and malformed-document rejection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._build import make_results_doc
from wattbench.schema import (
    ResultsSchemaError,
    load_results,
    validate_results,
    write_results,
)


def test_round_trip_validation(tmp_path: Path) -> None:
    doc = make_results_doc(throughput=123.0)
    path = tmp_path / "results.json"
    written = write_results(doc, path)
    assert written.cells[0].aggregates["throughput_per_s"].mean == pytest.approx(123.0)
    loaded = load_results(path)
    assert loaded.schema_version == "1.0"
    assert loaded.cells[0].cell_id == doc["cells"][0]["cell_id"]
    assert loaded.cells[0].per_repeat[0].alignment.offset_ms == pytest.approx(1.0)


def test_rejects_malformed_doc() -> None:
    doc = make_results_doc()
    doc["schema_version"] = "9.9"
    with pytest.raises(ResultsSchemaError, match="schema_version"):
        validate_results(doc)

    missing = make_results_doc()
    del missing["run"]
    with pytest.raises(ResultsSchemaError, match="run"):
        validate_results(missing)

    empty = make_results_doc()
    empty["cells"] = []
    with pytest.raises(ResultsSchemaError, match="cells"):
        validate_results(empty)

    extra = make_results_doc()
    extra["cells"][0]["unexpected"] = 1
    with pytest.raises(ResultsSchemaError, match="unexpected"):
        validate_results(extra)


def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ResultsSchemaError, match="not valid JSON"):
        load_results(path)


def test_written_document_is_json_serialisable(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    write_results(make_results_doc(), path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "1.0"
    assert isinstance(parsed["cells"], list)

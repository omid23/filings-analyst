from __future__ import annotations

import json
import shutil

import pytest

from filings_analyst.cli import main


def test_cli_index_search_eval(tmp_path, corpus, capsys):
    folder = tmp_path / "pdfs"
    folder.mkdir()
    shutil.copy(corpus[0].source, folder / "ACME_2023_10K.pdf")
    shutil.copy(corpus[2].source, folder / "Globex-Corp_2023_10K.pdf")
    index = str(tmp_path / "idx")

    main(["--index", index, "index", "--folder", str(folder), "--workers", "1"])
    stats = json.loads(capsys.readouterr().out)
    assert stats["documents"] == 2 and stats["chunks"] > 10

    main(["--index", index, "search", "Globex Corp 2023 statements of income net sales", "-k", "3"])
    out = capsys.readouterr().out
    assert "1. Globex-Corp_2023_10K p.7  Consolidated Statements of Income" in out

    with pytest.raises(SystemExit, match="--financebench"):
        main(["--index", index, "index"])
    with pytest.raises(SystemExit):
        main(["--index", index, "ask", "q"])  # --provider is required

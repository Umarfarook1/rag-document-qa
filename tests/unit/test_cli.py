import pytest

from rag_document_qa.cli import main


def test_cli_help_lists_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "ingest" in out
    assert "ask" in out
    assert "evals" in out


def test_cli_no_args_prints_hint(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    err = capsys.readouterr().err
    assert rc == 1
    assert "rag-document-qa" in err


def test_cli_evals_without_subcommand_prints_hint(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["evals"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "evals run" in err


def test_cli_ingest_no_inputs_returns_error(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["ingest"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no documents" in err


def test_cli_ingest_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--tickers" in out
    assert "--paths" in out
    assert "--embedder" in out


def test_cli_ask_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--top-k" in out
    assert "--no-llm" in out


def test_cli_evals_run_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["evals", "run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--golden" in out
    assert "--report" in out
    assert "--limit" in out
    assert "--badge-metric" in out

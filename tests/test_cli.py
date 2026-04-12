import os
from unittest.mock import patch

from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def test_list_indexers():
    result = runner.invoke(app, ["list-indexers"])
    assert result.exit_code == 0
    assert "Available Indexers" in result.stdout
    assert "faiss" in result.stdout


def test_setup():
    # Use a dummy model to avoid pulling anything
    result = runner.invoke(app, ["setup", "--model", "not-a-model"])
    assert result.exit_code == 0
    assert "Embenx Environment Check" in result.stdout


def test_cleanup_no_artifacts():
    result = runner.invoke(app, ["cleanup"])
    assert result.exit_code == 0
    assert "No artifacts found" in result.stdout or "Successfully removed" in result.stdout


def test_cleanup_with_error():
    # Mock os.remove to fail
    with (
        patch("os.remove", side_effect=Exception("Permission denied")),
        patch("glob.glob", return_value=["dummy.db"]),
    ):
        result = runner.invoke(app, ["cleanup"])
        assert result.exit_code == 0
        assert "Failed to remove" in result.stdout


def test_init_skill():
    # Cleanup any existing SKILL.md
    if os.path.exists("SKILL.md"):
        os.remove("SKILL.md")

    result = runner.invoke(app, ["init-skill"])
    assert result.exit_code == 0
    assert "Created SKILL.md successfully" in result.stdout
    assert os.path.exists("SKILL.md")


def test_benchmark_help():
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "Run Embenx benchmarks" in result.stdout


@patch("benchmark.run_benchmark")
def test_cli_benchmark_run(mock_run):
    result = runner.invoke(
        app, ["benchmark", "--dataset", "dummy", "--max-docs", "5", "--indexers", "faiss"]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()


@patch("benchmark.run_benchmark")
def test_cli_benchmark_custom_indexer(mock_run):
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--dataset",
            "dummy",
            "--custom-indexer",
            "examples/custom_indexer.py",
            "--indexers",
            "mymockindexer",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    # Verify custom_indexer_script was passed
    args, kwargs = mock_run.call_args
    assert kwargs["custom_indexer_script"] == "examples/custom_indexer.py"


def test_setup_with_pull():
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value.stdout = "other-model"
        result = runner.invoke(app, ["setup", "--model", "ollama/nomic-embed-text", "--pull"])
        assert result.exit_code == 0
        # Check if pull was attempted (it should be since nomic-embed-text is not in stdout)
        assert any("pull" in str(args) for args in mock_sub.call_args_list)


def test_setup_ollama_error():
    with patch("subprocess.run", side_effect=Exception("Ollama down")):
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert "Ollama error" in result.stdout


def test_list_indexers_command():
    result = runner.invoke(app, ["list-indexers"])
    assert result.exit_code == 0
    assert "faiss" in result.stdout

from typer.testing import CliRunner
from cli import app
import os
import shutil

runner = CliRunner()

def test_list_indexers():
    result = runner.invoke(app, ["list-indexers"])
    assert result.exit_code == 0
    assert "Available Indexers" in result.stdout
    assert "faiss" in result.stdout
    assert "duckdb" in result.stdout

def test_setup():
    # Use a dummy model to avoid pulling anything
    result = runner.invoke(app, ["setup", "--model", "not-a-model"])
    assert result.exit_code == 0
    assert "Embenx Environment Check" in result.stdout

def test_cleanup_no_artifacts():
    result = runner.invoke(app, ["cleanup"])
    assert result.exit_code == 0
    assert "No artifacts found" in result.stdout or "Successfully removed" in result.stdout

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

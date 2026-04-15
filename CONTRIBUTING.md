<!-- generated-by: gsd-doc-writer -->
# Contributing to Embenx 🚀

Thank you for your interest in contributing to Embenx! We welcome contributions from everyone.

## Development Setup

We use `uv` for dependency management. To set up your environment for development:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/adityak74/embenx.git
    cd embenx
    ```

2.  **Install development dependencies:**
    ```bash
    uv sync --all-extras --dev
    ```

3.  **Install pre-commit hooks:**
    ```bash
    uv run pre-commit install
    ```

See [GETTING-STARTED.md](GETTING-STARTED.md) for prerequisites and first-run instructions, and [DEVELOPMENT.md](DEVELOPMENT.md) (if available) for more details on local development setup.

## Coding Standards

Contributors must follow these standards:

- **Linting & Formatting**: We use `ruff`, `black`, and `isort`. You can run them manually with `uv run ruff check .`, `uv run black .`, and `uv run isort .`, or use the pre-commit hooks.
- **Type Hints**: Use Python type hints for all new functions and classes.
- **Commit Messages**: We recommend using clear, descriptive commit messages.
- **CI Enforcement**: All PRs must pass the linting and testing steps in GitHub Actions before being merged.

## Test Suites

Embenx has a comprehensive test suite using `pytest`.

- **Run all tests**:
  ```bash
  uv run pytest tests/
  ```
- **Run a specific test file**:
  ```bash
  uv run pytest tests/test_core.py
  ```
- **Coverage**: To check test coverage, run:
  ```bash
  uv run pytest --cov=embenx tests/
  ```

The test suite covers:
- **Core Logic**: Validation of the agentic memory layer and data structures (`test_core.py`, `test_data.py`).
- **Indexers**: Integration and unit tests for each of the 15+ vector backends (`test_indexers.py`, `test_faiss.py`, etc.).
- **CLI**: Verification of the `embenx` CLI tool (`test_cli.py`).
- **Advanced Features**: Hybrid search, reranking, and multimodal retrieval (`test_hybrid.py`, `test_multimodal.py`).

## Pull Request Guidelines

1.  **Fork** the repository and create a new branch: `git checkout -b feat/my-feature`.
2.  **Implement** your changes and add corresponding tests.
3.  **Ensure** all tests pass and your code is formatted correctly.
4.  **Submit** a Pull Request to the `main` branch.
5.  **Review**: A maintainer will review your PR. Address any feedback as requested.

## Issue Reporting

If you find a bug or have a feature request, please open an issue on GitHub:
- **Bug reports**: Include a clear description, steps to reproduce, and your environment (OS, Python version).
- **Feature requests**: Explain the use case and how the feature would benefit users.

## Release Flow

Embenx uses a structured release process managed by the `publish.sh` script and `bump-my-version`.

1.  **Version Bump**: Maintainers run `./publish.sh [patch|minor|major]` to bump the version in `pyproject.toml` and create a git tag.
2.  **Build**: The script uses `python3 -m build` to generate source and wheel distributions.
3.  **Test Upload**: The package is first uploaded to **TestPyPI** for verification.
4.  **Production Release**: After confirmation, the package is uploaded to **PyPI** using `twine`.

Maintainers should ensure that the CHANGELOG (if applicable) is updated and all CI checks are passing on `main` before initiating a release.

---

Built with ❤️ by the Embenx community.

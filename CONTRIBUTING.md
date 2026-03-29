# Contributing to Embenx 🚀

Thank you for your interest in contributing to Embenx! We welcome contributions from everyone.

## How to Contribute

### 1. Reporting Bugs
If you find a bug, please open an issue on GitHub with a clear description and steps to reproduce.

### 2. Suggesting Features
Have an idea for a new feature? Open an issue to discuss it before starting work.

### 3. Submitting Pull Requests
1.  **Fork** the repository and create a new branch: `git checkout -b feature/my-feature`.
2.  **Install** development dependencies: `uv sync`.
3.  **Implement** your changes.
4.  **Add tests** for your changes in the `tests/` directory.
5.  **Run tests** and ensure everything passes: `uv run pytest`.
6.  **Lint and format** your code:
    -   `uv run ruff check . --fix`
    -   `uv run black .`
    -   `uv run isort .`
7.  **Commit** your changes with a clear message.
8.  **Push** to your fork and **open a Pull Request** to the `main` branch.

## Development Setup

We use `uv` for dependency management. To set up your environment:

```bash
git clone https://github.com/adityak74/embenx.git
cd embenx
uv sync
```

## Adding a New Indexer

To add a new vector indexer:
1.  Create a new file in `indexers/` (e.g., `my_new_indexer.py`).
2.  Inherit from `BaseIndexer` and implement the required methods (`build_index`, `search`, `get_size`).
3.  Register your indexer in `indexers/__init__.py` within the `get_indexer_map()` function.
4.  Add unit tests in `tests/`.

## Code Style

We follow PEP 8 and use `black`, `ruff`, and `isort` for automated formatting and linting. Please ensure your code passes these checks before submitting a PR.

---

Built with ❤️ by the Embenx community.

# Contributing

Thanks for your interest in contributing.

1. Open an issue describing the change before starting significant work.
2. Fork the repo and create a feature branch.
3. Keep commits focused and write a clear description.
4. Ensure existing checks pass before opening a pull request. CI runs these four, so run them locally first:

```bash
pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
mypy src
pytest
```

`pytest` deselects the `edgar` and `live` markers by default. Those hit SEC EDGAR and the Anthropic API; run them explicitly with `pytest -m edgar` or `pytest -m live`.

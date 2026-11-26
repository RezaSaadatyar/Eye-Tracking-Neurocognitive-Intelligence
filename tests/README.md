# Software and Data Contract Tests

These tests use synthetic fixtures to verify data handling, alignment, model interfaces and evaluation calculations. They do not independently reproduce the original-data experiments.

Run the relevant tests in an environment with the root and applicable study dependencies:

```bash
python -m pytest tests/ -q
python -m pytest reading_comprehension_modelling/code/tests/ -q
```

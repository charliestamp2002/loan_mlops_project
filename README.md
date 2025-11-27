# Loan MLOps Project

An end-to-end MLOps-style project for a two-stage loan decision pipeline:

1. **Loan approval model** – predicts whether an application should be approved.
2. **Default risk model** – estimates probability of default for approved loans.
3. **Decision engine** – combines model outputs with business rules to output APPROVE / REJECT / MANUAL_REVIEW.

## Structure

- `data/` – raw and processed datasets.
- `loan_mlops/` – Python package with data, models, decision engine, and API.
- `scripts/` – command-line entry points.
- `tests/` – unit tests.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt


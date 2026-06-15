from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CURRENCY_EUR_RATES = {
    "EUR": 1.0,
    "USD": 1.17,
    "CZK": 24.4,
    "DKK": 7.47,
    "HUF": 366.0,
    "PLN": 4.24,
    "CHF": 0.92,
    "NOK": 10.95,
    "CAD": 1.6,
    "THB": 38.0,
}
SALARY_PERIOD_MULTIPLIERS = {
    "hour": 1800,
    "day": 225,
    "month": 12,
    "year": 1,
}
DEFAULT_MINIMUM_SALARY = 55000

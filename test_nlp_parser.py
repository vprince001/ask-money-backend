from datetime import datetime, timedelta

from nlp_parser import parse_expense


def test_parse_expense_spent_on_food():
    result = parse_expense("Spent 200 on food")
    assert result["amount"] == 200
    assert result["category"] == "food"
    assert result["date"] == datetime.now().date().isoformat()


def test_parse_expense_petrol_yesterday():
    result = parse_expense("I paid 500 for petrol yesterday")
    assert result["amount"] == 500
    assert result["category"] == "petrol"
    assert result["date"] == (datetime.now().date() - timedelta(days=1)).isoformat()


def test_parse_expense_groceries_rupees():
    result = parse_expense("Add 1200 rupees for groceries")
    assert result["amount"] == 1200
    assert result["category"] == "groceries"


def test_parse_expense_today_and_coffee():
    result = parse_expense("Coffee 150 today")
    assert result["amount"] == 150
    assert result["category"] == "food"
    assert result["date"] == datetime.now().date().isoformat()


def test_parse_expense_last_week_fuel():
    result = parse_expense("Fuel 300 last week")
    assert result["amount"] == 300
    assert result["category"] == "petrol"
    assert result["date"] == (datetime.now().date() - timedelta(days=7)).isoformat()


def test_parse_expense_missing_amount_raises():
    try:
        parse_expense("coffee and groceries")
        assert False, "Expected ValueError for missing amount"
    except ValueError as exc:
        assert "amount" in str(exc).lower()

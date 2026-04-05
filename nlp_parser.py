import re
from datetime import datetime, timedelta
from typing import Dict

try:
    import spacy
    _nlp = spacy.blank("en")
except Exception:
    _nlp = None

CATEGORY_KEYWORDS: Dict[str, list[str]] = {
    "food": ["food", "restaurant", "coffee", "lunch", "dinner"],
    "petrol": ["petrol", "fuel", "diesel"],
    "groceries": ["grocery", "groceries"],
    "clothes": ["clothes", "shopping"],
}

AMOUNT_PATTERN = re.compile(r"(?:₹|rs\.?\s*)?(?P<amount>[0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)

DATE_KEYWORDS = {
    "today": 0,
    "yesterday": -1,
    "last week": -7,
}


def _normalize_amount(raw_amount: str) -> float:
    normalized = raw_amount.replace(",", "")
    if normalized.count(".") > 1:
        raise ValueError("Invalid amount format")
    return float(normalized)


def _extract_amount(text: str) -> float:
    for match in AMOUNT_PATTERN.finditer(text):
        try:
            amount = _normalize_amount(match.group("amount"))
            if amount > 0:
                return amount
        except ValueError:
            continue
    raise ValueError("No valid amount found in text")


def _extract_category(text: str) -> str:
    normalized = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", normalized):
                return category
    return "other"


def _extract_date(text: str) -> str:
    normalized = text.lower()

    for phrase, offset in DATE_KEYWORDS.items():
        if phrase in normalized:
            return (datetime.now().date() + timedelta(days=offset)).isoformat()

    return datetime.now().date().isoformat()


def _extract_item_name(text: str, category: str) -> str:
    if _nlp is not None:
        doc = _nlp(text)
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        return " ".join(tokens).strip()

    return text.strip()


def parse_expense(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Expense text is required")

    amount = _extract_amount(text)
    if amount <= 0:
        raise ValueError("Amount must be greater than zero")

    category = _extract_category(text)
    date = _extract_date(text)
    item_name = _extract_item_name(text, category)

    confidence = 0.95 if category != "other" else 0.75

    return {
        "amount": amount,
        "category": category,
        "date": date,
        "item_name": item_name,
        "confidence": confidence,
        "original_text": text.strip(),
    }

import os
from openai import OpenAI

print("API KEY:", os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def parse_expense_with_llm(text: str) -> dict:
    prompt = f"""
You are an expense parser.

Extract structured data from the text.

Text: "{text}"

STRICT RULES:
- Return ONLY valid JSON (no explanation)
- item_name must be SHORT and CLEAN (e.g., "bread", not full sentence)
- Remove verbs, filler words, currency words
- category must be one of:
  ["food", "groceries", "transport", "shopping", "grooming", "bills", "entertainment", "other"]

CATEGORY RULES:

You must classify into one of these categories ONLY:
["groceries", "food", "transport", "shopping", "grooming", "bills", "entertainment", "health", "education", "other"]

DETAILED MAPPING:

GROCERIES:
- milk, bread, butter, eggs, fruits, vegetables, rice, wheat, flour, oil, sugar
- snacks, chocolates, biscuits, chips, juice
- any household consumables

FOOD (eating out):
- pizza, burger, sandwich, restaurant, cafe, swiggy, zomato, dining, lunch, dinner

TRANSPORT:
- uber, ola, taxi, auto, bus, metro, train, flight, fuel, petrol, diesel

SHOPPING:
- clothes, shoes, electronics, amazon, flipkart, accessories

GROOMING:
- haircut, salon, spa, shaving, cosmetics

BILLS:
- electricity, water, internet, recharge, mobile bill, rent

ENTERTAINMENT:
- movie, netflix, games, subscriptions

HEALTH:
- doctor, medicine, hospital, pharmacy

EDUCATION:
- books, course, fees, tuition

IMPORTANT RULES:
- Chocolates, snacks → groceries (NOT food)
- Only classify as transport if clear travel words exist
- If unsure → "other"

OUTPUT FORMAT:
{{
  "amount": number,
  "item_name": string,
  "category": string,
  "confidence": number (0 to 1)
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # fast + cheap
        messages=[
            {"role": "system", "content": "You are a precise financial data extractor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    try:
        import json
        return json.loads(content)
    except Exception:
        return {
            "amount": None,
            "item_name": text,
            "category": "other",
            "confidence": 0.3
        }
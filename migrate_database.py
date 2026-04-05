#!/usr/bin/env python3
"""
Database Migration Script for Enhanced Expense Tracker
Migrates existing data to new schema with categories system
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime, timezone
import uuid

# MongoDB connection
MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "test_database"

GLOBAL_CATEGORIES = [
    {"name": "Food", "emoji": "🍔", "scope": "global"},
    {"name": "Groceries", "emoji": "🛒", "scope": "global"},
    {"name": "Travel", "emoji": "🚕", "scope": "global"},
    {"name": "Rent", "emoji": "🏠", "scope": "global"},
    {"name": "Health", "emoji": "💊", "scope": "global"},
    {"name": "Entertainment", "emoji": "🎮", "scope": "global"},
    {"name": "Miscellaneous", "emoji": "📦", "scope": "global"},
]

CATEGORY_MAPPING = {
    "food": "Food",
    "groceries": "Groceries",
    "grocery": "Groceries",
    "travel": "Travel",
    "transport": "Travel",
    "rent": "Rent",
    "health": "Health",
    "medical": "Health",
    "entertainment": "Entertainment",
    "miscellaneous": "Miscellaneous",
    "misc": "Miscellaneous",
}

async def migrate():
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    
    print("🚀 Starting database migration...")
    
    # Step 1: Create global categories
    print("\n📁 Step 1: Creating global categories...")
    category_id_map = {}
    
    for cat in GLOBAL_CATEGORIES:
        existing = await db.categories.find_one({"name": cat["name"], "scope": "global"})
        if existing:
            category_id_map[cat["name"]] = existing["category_id"]
            print(f"   ✓ Category '{cat['name']}' already exists")
        else:
            category_id = f"cat_{uuid.uuid4().hex[:12]}"
            await db.categories.insert_one({
                "category_id": category_id,
                "name": cat["name"],
                "emoji": cat["emoji"],
                "scope": cat["scope"],
                "user_id": None,
                "group_id": None,
                "created_by": "system",
                "created_at": datetime.now(timezone.utc)
            })
            category_id_map[cat["name"]] = category_id
            print(f"   ✓ Created category '{cat['name']}' ({category_id})")
    
    # Step 2: Migrate existing expenses
    print("\n📝 Step 2: Migrating existing expenses...")
    expenses = await db.expenses.find({}, {"_id": 0}).to_list(10000)
    migrated_count = 0
    
    for expense in expenses:
        # Skip if already migrated
        if "category_id" in expense:
            continue
        
        # Map old category to new category_id
        old_category = expense.get("category", "miscellaneous").lower()
        category_name = CATEGORY_MAPPING.get(old_category, "Miscellaneous")
        category_id = category_id_map.get(category_name)
        
        if not category_id:
            print(f"   ⚠️  Warning: No category found for '{old_category}', using Miscellaneous")
            category_id = category_id_map["Miscellaneous"]
        
        # Update expense
        update_data = {
            "category_id": category_id,
            "item_name": None,  # New field
            "added_by_picture": None  # New field
        }
        
        # Remove old fields
        unset_data = {
            "category": "",
            "emoji": ""
        }
        
        await db.expenses.update_one(
            {"expense_id": expense["expense_id"]},
            {
                "$set": update_data,
                "$unset": unset_data
            }
        )
        migrated_count += 1
    
    print(f"   ✓ Migrated {migrated_count} expenses")
    
    # Step 3: Create indexes
    print("\n🔍 Step 3: Creating database indexes...")
    
    # Categories indexes
    await db.categories.create_index("category_id", unique=True)
    await db.categories.create_index([("scope", 1), ("user_id", 1)])
    await db.categories.create_index([("scope", 1), ("group_id", 1)])
    print("   ✓ Category indexes created")
    
    # Expenses indexes
    await db.expenses.create_index("category_id")
    await db.expenses.create_index("user_id")
    await db.expenses.create_index("group_id")
    await db.expenses.create_index("date")
    print("   ✓ Expense indexes created")
    
    # Groups indexes
    await db.groups.create_index("group_id", unique=True)
    await db.groups.create_index("created_by")
    print("   ✓ Group indexes created")
    
    print("\n✅ Migration completed successfully!")
    print(f"\nSummary:")
    print(f"  - Global categories: {len(GLOBAL_CATEGORIES)}")
    print(f"  - Expenses migrated: {migrated_count}")
    print(f"  - Indexes created: 9")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(migrate())

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Cookie, Response, Request, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import httpx
from nlp_parser import parse_expense as parse_expense_nlp
import json
import base64
import re
from jose import jwt
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from services.llm_parser import parse_expense_with_llm
from openai import OpenAI
import tempfile


# Auth configuration
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'replace-this-secret')
JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')
JWT_EXPIRES_SECONDS = int(os.environ.get('JWT_EXPIRES_SECONDS', 7 * 24 * 60 * 60))
SECURE_COOKIES = os.environ.get('SECURE_COOKIES', '0') == '1'

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= MODELS =============

class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime

class Category(BaseModel):
    category_id: str
    name: str
    emoji: str
    scope: str  # 'global', 'private', 'group'
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    created_by: str
    created_at: datetime

class UserSession(BaseModel):
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime

class Group(BaseModel):
    group_id: str
    name: str
    created_by: str  # user_id
    created_at: datetime

class GroupMember(BaseModel):
    member_id: str
    group_id: str
    user_id: str
    joined_at: datetime

class Expense(BaseModel):
    expense_id: str
    amount: float
    category: str
    emoji: str
    date: datetime
    user_id: str
    group_id: Optional[str] = None
    added_by_name: str
    notes: Optional[str] = None  # Original voice transcription
    created_at: datetime

# Request/Response Models
class SessionCreate(BaseModel):
    session_id: str

class GoogleAuthRequest(BaseModel):
    id_token: str

class SessionResponse(BaseModel):
    session_token: str
    user: User

class ExpenseParseRequest(BaseModel):
    text: str

class ExpenseParsedResponse(BaseModel):
    amount: float
    category: str
    emoji: str
    date: str
    confidence: float
    original_text: str

class ExpenseCreate(BaseModel):
    amount: float
    category: str
    item_name: Optional[str] = None
    date: str
    group_id: Optional[str] = None
    notes: Optional[str] = None

class GroupCreate(BaseModel):
    name: str

class GroupInvite(BaseModel):
    user_email: str

class ExpenseUpdate(BaseModel):
    amount: Optional[float] = None
    category_id: Optional[str] = None
    item_name: Optional[str] = None
    date: Optional[str] = None
    group_id: Optional[str] = None
    notes: Optional[str] = None

class GroupBudgetCreate(BaseModel):
    group_id: str
    total_budget: float
    start_date: str
    end_date: str
    is_recurring: bool = False

class GroupBudgetUpdate(BaseModel):
    total_budget: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_recurring: Optional[bool] = None

class CategoryBudgetCreate(BaseModel):
    category_id: str
    allocated_amount: float

class CategoryBudgetUpdate(BaseModel):
    allocated_amount: float

class AudioTranscribeRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = None  # e.g., "en", "hi" - recommended for accuracy

class AnalyticsSummary(BaseModel):
    total_spend: float
    category_breakdown: dict
    expense_count: int
    period: str

# ============= CATEGORY MAPPING =============

CATEGORY_MAP = {
    # GROCERIES
    "bread": ("Groceries", "🛒"),
    "milk": ("Groceries", "🥛"),
    "vegetables": ("Groceries", "🥦"),
    "fruits": ("Groceries", "🍎"),
    "groceries": ("Groceries", "🛒"),
    "grocery": ("Groceries", "🛒"),

    # FOOD
    "pizza": ("Food", "🍕"),
    "burger": ("Food", "🍔"),
    "dinner": ("Food", "🍽️"),
    "dinner": ("Food", "🍔"),
    "lunch": ("Food", "🍔"),
    "breakfast": ("Food", "🍔"),
    "burger": ("Food", "🍔"),
    "pizza": ("Food", "🍔"),

    # GROOMING
    "haircut": ("Grooming", "💇"),

    # Travel
    "travel": ("Travel", "🚕"),
    "transport": ("Travel", "🚕"),
    "uber": ("Travel", "🚕"),
    "ola": ("Travel", "🚕"),
    "petrol": ("Travel", "🚕"),
    "fuel": ("Travel", "🚕"),

    "rent": ("Rent", "🏠"),

    # Health
    "health": ("Health", "💊"),
    "medical": ("Health", "💊"),
    "medicine": ("Health", "💊"),
    "doctor": ("Health", "💊"),

    # Entertainment
    "entertainment": ("Entertainment", "🎮"),
    "movie": ("Entertainment", "🎮"),
    "game": ("Entertainment", "🎮"),

    # Miscellaneous
    "miscellaneous": ("Miscellaneous", "📦"),
    "misc": ("Miscellaneous", "📦"),
}

def get_category_and_emoji(category_text: str) -> tuple:
    """Map category text to standard category and emoji"""
    category_lower = category_text.lower().strip()
    if category_lower in CATEGORY_MAP:
        return CATEGORY_MAP[category_lower]
    # Default to Miscellaneous
    return ("Miscellaneous", "📦")

# ============= AUTH HELPERS =============

async def get_current_user(
    request: Request,
    session_token: Optional[str] = Cookie(None)
) -> User:
    """Get current authenticated user from session token (cookie or header)"""
    # Try cookie first, then Authorization header
    token = session_token
    if not token:
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find session
    session_doc = await db.user_sessions.find_one(
        {"session_token": token},
        {"_id": 0}
    )
    
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check expiry
    expires_at = session_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Get user
    user_doc = await db.users.find_one(
        {"user_id": session_doc["user_id"]},
        {"_id": 0}
    )
    
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    return User(**user_doc)

# ============= AUTH ROUTES =============

@api_router.post("/auth/google", response_model=SessionResponse)
async def google_login(data: GoogleAuthRequest, response: Response):
    """Verify Google ID token and create a local session"""
    if not GOOGLE_CLIENT_ID or not JWT_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Google auth configuration is missing")

    try:
        idinfo = google_id_token.verify_oauth2_token(
            data.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        if idinfo.get("iss") not in ["accounts.google.com", "https://accounts.google.com"]:
            raise HTTPException(status_code=401, detail="Invalid Google issuer")

        email = idinfo.get("email")
        if not email:
            raise HTTPException(status_code=401, detail="Google token did not contain an email")

        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user_doc = await db.users.find_one({"email": email}, {"_id": 0})

        if user_doc:
            user_id = user_doc["user_id"]
            await db.users.update_one(
                {"user_id": user_id},
                {"$set": {
                    "name": idinfo.get("name"),
                    "picture": idinfo.get("picture")
                }}
            )
        else:
            user_doc = {
                "user_id": user_id,
                "email": email,
                "name": idinfo.get("name", email.split("@")[0]),
                "picture": idinfo.get("picture"),
                "created_at": datetime.now(timezone.utc)
            }
            await db.users.insert_one(user_doc)
            # 🔥 Create default categories for new user
            default_categories = [
                ("Groceries", "🛒"),
                ("Food", "🍔"),
                ("Travel", "🚕"),
                ("Shopping", "🛍️"),
                ("Grooming", "💇"),
                ("Bills", "📄"),
                ("Entertainment", "🎮"),
                ("Health", "💊"),
                ("Rent", "🏠"),
                ("Miscellaneous", "📦"),
            ]

            for name, emoji in default_categories:
                await db.categories.insert_one({
                    "category_id": f"cat_{uuid.uuid4().hex[:12]}",
                    "name": name,
                    "emoji": emoji,
                    "scope": "private",
                    "user_id": user_id,
                    "created_by": user_id,
                    "created_at": datetime.now(timezone.utc)
                })

        # Create JWT session token and persist it locally
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=JWT_EXPIRES_SECONDS)
        session_token = jwt.encode(
            {
                "user_id": user_id,
                "email": email,
                "exp": int(expires_at.timestamp())
            },
            JWT_SECRET_KEY,
            algorithm=JWT_ALGORITHM
        )

        await db.user_sessions.update_one(
            {"user_id": user_id},
            {"$set": {
                "session_token": session_token,
                "expires_at": expires_at,
                "created_at": datetime.now(timezone.utc)
            }},
            upsert=True
        )

        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=SECURE_COOKIES,
            samesite="lax",
            path="/",
            max_age=JWT_EXPIRES_SECONDS
        )

        user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
        return {
            "session_token": session_token,
            "user": User(**user)
        }
    except ValueError as e:
        logger.error(f"Google auth verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid Google login token")
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user"""
    return current_user

@api_router.post("/auth/logout")
async def logout(response: Response, session_token: Optional[str] = Cookie(None)):
    """Logout user"""
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    
    response.delete_cookie(
        key="session_token",
        path="/",
        samesite="none",
        secure=True
    )
    
    return {"message": "Logged out successfully"}


# ============= CATEGORY ROUTES =============

@api_router.post("/categories")
async def create_category(
    name: str,
    emoji: str,
    scope: str,
    group_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Create a custom category"""
    if scope not in ["private", "group"]:
        raise HTTPException(status_code=400, detail="Scope must be 'private' or 'group'")
    
    if scope == "group" and not group_id:
        raise HTTPException(status_code=400, detail="group_id required for group category")
    
    if scope == "group":
        member = await db.group_members.find_one({"group_id": group_id, "user_id": current_user.user_id})
        if not member:
            raise HTTPException(status_code=403, detail="Not a member of this group")
    
    category_id = f"cat_{uuid.uuid4().hex[:12]}"
    category_doc = {
        "category_id": category_id,
        "name": name,
        "emoji": emoji,
        "scope": scope,
        "user_id": current_user.user_id if scope == "private" else None,
        "group_id": group_id if scope == "group" else None,
        "created_by": current_user.user_id,
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.categories.insert_one(category_doc)
    # Return the created category without MongoDB _id
    created_category = await db.categories.find_one({"category_id": category_id}, {"_id": 0})
    return created_category

@api_router.get("/categories")
async def list_categories(group_id: Optional[str] = None, current_user: User = Depends(get_current_user)):
    """List available categories"""
    query_list = [
        {"scope": "global"},
        {"scope": "private", "user_id": current_user.user_id}
    ]
    
    if group_id:
        member = await db.group_members.find_one({"group_id": group_id, "user_id": current_user.user_id})
        if member:
            query_list.append({"scope": "group", "group_id": group_id})
    
    categories = await db.categories.find({"$or": query_list}, {"_id": 0}).to_list(1000)
    return categories

@api_router.put("/categories/{category_id}")
async def update_category(
    category_id: str,
    name: Optional[str] = None,
    emoji: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Update a category"""
    category = await db.categories.find_one({"category_id": category_id}, {"_id": 0})
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    if category["scope"] == "global":
        raise HTTPException(status_code=403, detail="Cannot edit global categories")
    if category["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_data = {}
    if name:
        update_data["name"] = name
    if emoji:
        update_data["emoji"] = emoji
    
    if update_data:
        await db.categories.update_one({"category_id": category_id}, {"$set": update_data})
    
    return await db.categories.find_one({"category_id": category_id}, {"_id": 0})

@api_router.delete("/categories/{category_id}")
async def delete_category(category_id: str, current_user: User = Depends(get_current_user)):
    """Delete a category"""
    category = await db.categories.find_one({"category_id": category_id}, {"_id": 0})
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    if category["scope"] == "global":
        raise HTTPException(status_code=403, detail="Cannot delete global categories")
    if category["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    expense_count = await db.expenses.count_documents({"category_id": category_id})
    if expense_count > 0:
        raise HTTPException(status_code=400, detail=f"Cannot delete: {expense_count} expenses use this category")
    
    await db.categories.delete_one({"category_id": category_id})
    return {"message": "Category deleted successfully"}

# ============= AI PARSING ROUTES =============

@api_router.post("/expenses/parse")
async def parse_expense_api(data: ExpenseParseRequest, current_user: User = Depends(get_current_user)):
    """Parse natural language expense using a local offline NLP parser."""
    try:
        parsed = parse_expense_nlp(data.text)
        logger.info(f"Rule parser: {parsed}")
        
        llm_result = None
        
        if parsed.get("confidence", 0) < 0.8 or not parsed.get("category"):
            try:
                llm_result = parse_expense_with_llm(data.text)
                logger.info(f"LLM parser: {llm_result}")
            except Exception as e:
                logger.error(f"LLM parsing failed: {e}")
                
        if llm_result and llm_result.get("confidence", 0) > parsed.get("confidence", 0):
            parsed = llm_result

            # CLEAN ITEM NAME
            if parsed.get("item_name"):
                item = parsed["item_name"].lower()
            
            # remove common noise
            for word in ["i bought", "i spent", "rupees", "rs", "for"]:
                item = item.replace(word, "")
                
            parsed["item_name"] = item.strip().capitalize()

            # Override category using item_name if possible
            item = parsed.get("item_name", "").lower()
            
            # Strong override rules
            if any(x in item for x in ["bread", "milk", "chocolate", "snack", "fruit", "vegetable"]):
                parsed["category"] = "groceries"
                
            elif any(x in item for x in ["pizza", "burger", "restaurant", "food"]):
                parsed["category"] = "food"

            elif any(x in item for x in ["uber", "ola", "taxi", "petrol", "fuel"]):
                parsed["category"] = "travel"

            elif any(x in item for x in ["haircut", "salon"]):
                parsed["category"] = "grooming"

            # Ensure date exists
            if "date" not in parsed or not parsed["date"]:
                parsed["date"] = datetime.utcnow().strftime("%Y-%m-%d")

        # Always run this outsine LLM check to ensure we have a category and emoji
        category_name, emoji = get_category_and_emoji(parsed.get("category", "misc"))
        return {
            "amount": parsed["amount"],
            "category": category_name,
            "emoji": emoji,
            "date": parsed.get("date") or datetime.utcnow().strftime("%Y-%m-%d"),
            "item_name": parsed.get("item_name"),
            "confidence": parsed.get("confidence", 1.0),
            "original_text": parsed.get("original_text", data.text),
        }
    except ValueError as e:
        logger.error(f"Parse validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Parse error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse expense: {str(e)}")

# ============= USER PREFERENCES =============

@api_router.get("/user/preferences")
async def get_preferences(current_user: User = Depends(get_current_user)):
    """Get user preferences"""
    prefs = await db.user_preferences.find_one({"user_id": current_user.user_id}, {"_id": 0})
    if not prefs:
        return {"user_id": current_user.user_id, "language": "en"}
    return prefs

@api_router.put("/user/preferences")
async def update_preferences(data: dict, current_user: User = Depends(get_current_user)):
    """Update user preferences"""
    allowed_fields = {"language"}
    update = {k: v for k, v in data.items() if k in allowed_fields}
    if not update:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    
    await db.user_preferences.update_one(
        {"user_id": current_user.user_id},
        {"$set": update, "$setOnInsert": {"user_id": current_user.user_id}},
        upsert=True
    )
    return await db.user_preferences.find_one({"user_id": current_user.user_id}, {"_id": 0})

# ============= ARCHIVED GROUPS =============

@api_router.get("/expenses/archived-groups")
async def list_archived_groups(current_user: User = Depends(get_current_user)):
    """List archived group summaries for the user"""
    pipeline = [
        {"$match": {"user_id": current_user.user_id, "is_group_deleted": True}},
        {"$group": {
            "_id": "$original_group_id",
            "group_name": {"$first": "$original_group_name"},
            "total_spent": {"$sum": "$amount"},
            "expense_count": {"$sum": 1},
            "last_expense_date": {"$max": "$date"},
        }},
        {"$sort": {"last_expense_date": -1}},
    ]
    results = await db.expenses.aggregate(pipeline).to_list(100)
    return [
        {
            "original_group_id": r["_id"],
            "group_name": r["group_name"],
            "total_spent": round(r["total_spent"], 2),
            "expense_count": r["expense_count"],
            "last_expense_date": r["last_expense_date"],
        }
        for r in results
    ]

# ============= VOICE TRANSCRIPTION =============

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@api_router.post("/voice/transcribe")
async def transcribe_audio(data: AudioTranscribeRequest, current_user: User = Depends(get_current_user)):
    try:
        # Decode base64
        audio_bytes = base64.b64decode(data.audio_base64)

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Send to OpenAI
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                language=data.language or "en"
            )

        return {"text": transcript.text}

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to transcribe audio: {str(e)}")


@api_router.post("/transcribe")
async def transcribe_audio_file_upload(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Transcribe audio using local Faster Whisper model (multipart file upload)"""
    from transcription_service import (
        transcribe_audio_file as do_transcribe, validate_file_size,
        save_temp_audio, cleanup_temp_file,
    )

    temp_path = None
    try:
        # Read file content
        audio_bytes = await file.read()

        # Validate file size
        if not validate_file_size(len(audio_bytes)):
            raise HTTPException(status_code=400, detail="Audio file too large (max 25MB)")

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Determine extension from filename
        ext = ".m4a"
        if file.filename:
            if "." in file.filename:
                ext = "." + file.filename.rsplit(".", 1)[1].lower()

        # Save temporarily
        temp_path = save_temp_audio(audio_bytes, ext)

        # Transcribe
        result = do_transcribe(temp_path)
        logger.info(f"Transcription successful: {result['text']}")

        return {
            "text": result["text"],
            "language": result.get("language"),
            "duration": result.get("duration"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to transcribe audio: {str(e)}")
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)

# ============= EXPENSE ROUTES =============

@api_router.post("/expenses")
async def create_expense(data: ExpenseCreate, current_user: User = Depends(get_current_user)):
    """Create a new expense with category validation"""
    expense_id = f"expense_{uuid.uuid4().hex[:12]}"
    
    # Parse date
    expense_date = datetime.fromisoformat(data.date) if isinstance(data.date, str) else data.date

    # Try to find category by name
    category = await db.categories.find_one({
        "name": {"$regex": f"^{data.category}$", "$options": "i"},
        "user_id": current_user.user_id
    }, {"_id": 0})
    
    # If not found → fallback to Miscellaneous
    if not category:
        category = await db.categories.find_one({
            "name": "Miscellaneous",
            "user_id": current_user.user_id
        }, {"_id": 0})
        
    # If still not found → create it automatically
    if not category:
        category_id = f"cat_{uuid.uuid4().hex[:12]}"
        category = {
            "category_id": category_id,
            "name": data.category or "Miscellaneous",
            "emoji": "📦",
            "scope": "private",
            "user_id": current_user.user_id,
            "created_by": current_user.user_id,
            "created_at": datetime.now(timezone.utc)
        }
        await db.categories.insert_one(category)

    category_id = category["category_id"]
    
    # Validate group membership if group expense
    if data.group_id:
        membership = await db.group_members.find_one({
            "group_id": data.group_id,
            "user_id": current_user.user_id
        })
        if not membership:
            raise HTTPException(status_code=403, detail="You are not a member of this group")
    
    # Validate category scope
    if data.group_id:
        # Group expense
        if category["scope"] == "private":
            raise HTTPException(status_code=400, detail="Cannot use private category in group expense")
        if category["scope"] == "group" and category["group_id"] != data.group_id:
            raise HTTPException(status_code=400, detail="Category belongs to different group")
    else:
        # Private expense
        if category["scope"] == "group":
            raise HTTPException(status_code=400, detail="Cannot use group category in private expense")
        if category["scope"] == "private" and category["user_id"] != current_user.user_id:
            raise HTTPException(status_code=403, detail="Cannot use another user's private category")
    
    expense_doc = {
        "expense_id": expense_id,
        "amount": data.amount,
        "category_id": category_id,
        "item_name": data.item_name,
        "date": expense_date,
        "user_id": current_user.user_id,
        "group_id": data.group_id,
        "added_by_name": current_user.name,
        "added_by_picture": current_user.picture,
        "notes": data.notes,
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.expenses.insert_one(expense_doc)
    
    # Return expense with category info populated (without MongoDB _id)
    created_expense = await db.expenses.find_one({"expense_id": expense_id}, {"_id": 0})
    created_expense["category"] = {
        "category_id": category["category_id"],
        "name": category["name"],
        "emoji": category.get("emoji", "📦"),
    }
    return created_expense

@api_router.get("/expenses")
async def list_expenses(
    group_id: Optional[str] = None,
    show_archived: bool = False,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """List expenses (personal or group) with category populated"""
    query = {"user_id": current_user.user_id}
    
    if group_id:
        # Verify user is member of group
        member = await db.group_members.find_one({
            "group_id": group_id,
            "user_id": current_user.user_id
        })
        if not member:
            raise HTTPException(status_code=403, detail="Not a member of this group")
        query = {"group_id": group_id}
    elif show_archived:
        # Archived group expenses only
        query["is_group_deleted"] = True
    else:
        # Personal expenses only (exclude archived)
        query["group_id"] = None
        query["is_group_deleted"] = {"$ne": True}
    
    expenses = await db.expenses.find(query, {"_id": 0}).sort("date", -1).limit(limit).to_list(limit)
    
    # Populate category for each expense (COMPATIBILITY LAYER)
    for expense in expenses:
        if "category_id" in expense:
            category = await db.categories.find_one(
                {"category_id": expense["category_id"]},
                {"_id": 0}
            )
            if category:
                expense["category"] = category["name"]
                expense["emoji"] = category["emoji"]
            else:
                expense["category"] = "Unknown"
                expense["emoji"] = "❓"
        # Ensure old fields exist for backward compatibility
        if "category" not in expense:
            expense["category"] = "Miscellaneous"
            expense["emoji"] = "📦"
    
    return expenses

@api_router.delete("/expenses/{expense_id}")
async def delete_expense(expense_id: str, current_user: User = Depends(get_current_user)):
    """Delete an expense"""
    # Verify ownership
    expense = await db.expenses.find_one({"expense_id": expense_id, "user_id": current_user.user_id})
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    
    await db.expenses.delete_one({"expense_id": expense_id})
    
    return {"message": "Expense deleted successfully"}

@api_router.patch("/expenses/{expense_id}")
async def update_expense(expense_id: str, data: ExpenseUpdate, current_user: User = Depends(get_current_user)):
    """Update an expense (creator only)"""
    expense = await db.expenses.find_one({"expense_id": expense_id}, {"_id": 0})
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    
    # Only creator can edit
    if expense["user_id"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Only the creator can edit this expense")
    
    update_fields = {}
    
    # Determine final group_id (might be changing)
    final_group_id = data.group_id if data.group_id is not None else expense.get("group_id")
    # Allow explicitly setting group_id to empty string to make private
    if data.group_id == "":
        final_group_id = None
        update_fields["group_id"] = None
    elif data.group_id is not None:
        # Validate group membership
        membership = await db.group_members.find_one({
            "group_id": data.group_id,
            "user_id": current_user.user_id
        })
        if not membership:
            raise HTTPException(status_code=403, detail="You are not a member of this group")
        update_fields["group_id"] = data.group_id
    
    # Determine category_id for scope validation
    if data.category_id:
        category = await db.categories.find_one({"category_id": data.category_id}, {"_id": 0})
        if not category:
            raise HTTPException(status_code=404, detail="Category not found")
        
        # Validate category scope
        if final_group_id:
            if category["scope"] == "private":
                raise HTTPException(status_code=400, detail="Cannot use private category in group expense")
            if category["scope"] == "group" and category.get("group_id") != final_group_id:
                raise HTTPException(status_code=400, detail="Category belongs to different group")
        else:
            if category["scope"] == "group":
                raise HTTPException(status_code=400, detail="Cannot use group category in private expense")
            if category["scope"] == "private" and category.get("user_id") != current_user.user_id:
                raise HTTPException(status_code=403, detail="Cannot use another user's private category")
        
        update_fields["category_id"] = data.category_id
    
    if data.amount is not None:
        if data.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")
        update_fields["amount"] = data.amount
    
    if data.item_name is not None:
        update_fields["item_name"] = data.item_name
    
    if data.date is not None:
        update_fields["date"] = datetime.fromisoformat(data.date)
    
    if data.notes is not None:
        update_fields["notes"] = data.notes
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    await db.expenses.update_one(
        {"expense_id": expense_id},
        {"$set": update_fields}
    )
    
    # Return updated expense with category populated
    updated = await db.expenses.find_one({"expense_id": expense_id}, {"_id": 0})
    if "category_id" in updated and updated["category_id"]:
        cat = await db.categories.find_one({"category_id": updated["category_id"]}, {"_id": 0})
        if cat:
            updated["category"] = cat["name"]
            updated["emoji"] = cat["emoji"]
    
    return updated

# ============= GROUP ROUTES =============

@api_router.post("/groups", response_model=Group)
async def create_group(data: GroupCreate, current_user: User = Depends(get_current_user)):
    """Create a new expense group"""
    group_id = f"group_{uuid.uuid4().hex[:12]}"
    
    group_doc = {
        "group_id": group_id,
        "name": data.name,
        "created_by": current_user.user_id,
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.groups.insert_one(group_doc)
    
    # Add creator as member
    await db.group_members.insert_one({
        "member_id": f"member_{uuid.uuid4().hex[:12]}",
        "group_id": group_id,
        "user_id": current_user.user_id,
        "joined_at": datetime.now(timezone.utc)
    })
    
    return Group(**group_doc)

@api_router.get("/groups", response_model=List[Group])
async def list_groups(current_user: User = Depends(get_current_user)):
    """List user's groups"""
    # Find groups where user is a member
    memberships = await db.group_members.find({"user_id": current_user.user_id}, {"_id": 0}).to_list(100)
    group_ids = [m["group_id"] for m in memberships]
    
    groups = await db.groups.find({"group_id": {"$in": group_ids}}, {"_id": 0}).to_list(100)
    
    return [Group(**g) for g in groups]

@api_router.post("/groups/{group_id}/members")
async def add_group_member(group_id: str, data: GroupInvite, current_user: User = Depends(get_current_user)):
    """Add member to group"""
    # Verify user is creator or member
    membership = await db.group_members.find_one({
        "group_id": group_id,
        "user_id": current_user.user_id
    })
    
    if not membership:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Find user by email
    invited_user = await db.users.find_one({"email": data.user_email}, {"_id": 0})
    if not invited_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if already member
    existing = await db.group_members.find_one({
        "group_id": group_id,
        "user_id": invited_user["user_id"]
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="User already a member")
    
    # Add member
    await db.group_members.insert_one({
        "member_id": f"member_{uuid.uuid4().hex[:12]}",
        "group_id": group_id,
        "user_id": invited_user["user_id"],
        "joined_at": datetime.now(timezone.utc)
    })
    
    return {"message": f"Added {data.user_email} to group"}

@api_router.put("/groups/{group_id}")
async def update_group(group_id: str, name: str, current_user: User = Depends(get_current_user)):
    """Update group name (members can edit)"""
    # Verify membership
    member = await db.group_members.find_one({"group_id": group_id, "user_id": current_user.user_id})
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    # Update group name
    await db.groups.update_one(
        {"group_id": group_id},
        {"$set": {"name": name}}
    )
    
    return await db.groups.find_one({"group_id": group_id}, {"_id": 0})

@api_router.delete("/groups/{group_id}")
async def delete_group(group_id: str, current_user: User = Depends(get_current_user)):
    """Delete group (creator only) - archives expenses with group context"""
    group = await db.groups.find_one({"group_id": group_id}, {"_id": 0})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Only creator can delete
    if group["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Only group creator can delete")
    
    group_name = group["name"]
    
    # Archive expenses: preserve group context for history
    await db.expenses.update_many(
        {"group_id": group_id},
        {"$set": {
            "group_id": None,
            "original_group_id": group_id,
            "original_group_name": group_name,
            "is_group_deleted": True,
        }}
    )
    
    # Delete budgets and allocations for this group
    budget_ids = []
    async for b in db.group_budgets.find({"group_id": group_id}, {"budget_id": 1}):
        budget_ids.append(b["budget_id"])
    if budget_ids:
        await db.category_budgets.delete_many({"budget_id": {"$in": budget_ids}})
        await db.group_budgets.delete_many({"group_id": group_id})
    
    # Delete all memberships
    await db.group_members.delete_many({"group_id": group_id})
    
    # Delete group-specific categories
    await db.categories.delete_many({"group_id": group_id, "scope": "group"})
    
    # Delete group
    await db.groups.delete_one({"group_id": group_id})
    
    return {"message": "Group deleted successfully"}

@api_router.get("/groups/{group_id}/members")
async def list_group_members(group_id: str, current_user: User = Depends(get_current_user)):
    """List members of a group"""
    # Verify membership
    member = await db.group_members.find_one({"group_id": group_id, "user_id": current_user.user_id})
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    memberships = await db.group_members.find({"group_id": group_id}, {"_id": 0}).to_list(100)
    members = []
    for m in memberships:
        user_doc = await db.users.find_one({"user_id": m["user_id"]}, {"_id": 0})
        if user_doc:
            members.append({
                "user_id": user_doc["user_id"],
                "name": user_doc["name"],
                "email": user_doc["email"],
                "picture": user_doc.get("picture"),
                "joined_at": m["joined_at"]
            })
    return members

# ============= BUDGET ROUTES =============

@api_router.post("/budgets")
async def create_budget(data: GroupBudgetCreate, current_user: User = Depends(get_current_user)):
    """Create a group budget (creator/admin only)"""
    group = await db.groups.find_one({"group_id": data.group_id}, {"_id": 0})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    if group["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Only the group creator can set budgets")
    
    if data.total_budget <= 0:
        raise HTTPException(status_code=400, detail="Total budget must be positive")
    
    start_date = datetime.fromisoformat(data.start_date)
    end_date = datetime.fromisoformat(data.end_date)
    if end_date <= start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")
    
    # Check for overlapping active budget
    existing = await db.group_budgets.find_one({
        "group_id": data.group_id,
        "start_date": {"$lte": end_date},
        "end_date": {"$gte": start_date}
    })
    if existing:
        raise HTTPException(status_code=400, detail="An overlapping budget already exists for this period")
    
    budget_id = f"budget_{uuid.uuid4().hex[:12]}"
    budget_doc = {
        "budget_id": budget_id,
        "group_id": data.group_id,
        "total_budget": round(data.total_budget, 2),
        "start_date": start_date,
        "end_date": end_date,
        "is_recurring": data.is_recurring,
        "created_by": current_user.user_id,
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.group_budgets.insert_one(budget_doc)
    created = await db.group_budgets.find_one({"budget_id": budget_id}, {"_id": 0})
    return created

@api_router.get("/budgets")
async def list_budgets(group_id: str, current_user: User = Depends(get_current_user)):
    """List budgets for a group"""
    member = await db.group_members.find_one({"group_id": group_id, "user_id": current_user.user_id})
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    budgets = await db.group_budgets.find(
        {"group_id": group_id},
        {"_id": 0}
    ).sort("start_date", -1).to_list(50)
    
    # Check for recurring auto-creation
    for budget in budgets:
        end_date = budget["end_date"]
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        if budget.get("is_recurring") and end_date < datetime.now(timezone.utc):
            # Check if next cycle already exists
            start_date = budget["start_date"]
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
                
            duration = end_date - start_date
            new_start = end_date
            new_end = new_start + duration
            existing_next = await db.group_budgets.find_one({
                "group_id": group_id,
                "start_date": {"$gte": new_start}
            })
            if not existing_next:
                # Auto-create next cycle
                new_budget_id = f"budget_{uuid.uuid4().hex[:12]}"
                new_budget = {
                    "budget_id": new_budget_id,
                    "group_id": group_id,
                    "total_budget": budget["total_budget"],
                    "start_date": new_start,
                    "end_date": new_end,
                    "is_recurring": True,
                    "created_by": budget["created_by"],
                    "created_at": datetime.now(timezone.utc)
                }
                await db.group_budgets.insert_one(new_budget)
                
                # Copy category allocations
                old_allocs = await db.category_budgets.find(
                    {"budget_id": budget["budget_id"]},
                    {"_id": 0}
                ).to_list(100)
                for alloc in old_allocs:
                    await db.category_budgets.insert_one({
                        "alloc_id": f"alloc_{uuid.uuid4().hex[:12]}",
                        "budget_id": new_budget_id,
                        "category_id": alloc["category_id"],
                        "allocated_amount": alloc["allocated_amount"],
                        "created_at": datetime.now(timezone.utc)
                    })
                
                new_created = await db.group_budgets.find_one({"budget_id": new_budget_id}, {"_id": 0})
                budgets.insert(0, new_created)
    
    return budgets

@api_router.get("/budgets/{budget_id}/summary")
async def get_budget_summary(budget_id: str, current_user: User = Depends(get_current_user)):
    """Get detailed budget summary with spending breakdown"""
    budget = await db.group_budgets.find_one({"budget_id": budget_id}, {"_id": 0})
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    member = await db.group_members.find_one({
        "group_id": budget["group_id"],
        "user_id": current_user.user_id
    })
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this group")
    
    # Get category allocations
    allocations = await db.category_budgets.find(
        {"budget_id": budget_id},
        {"_id": 0}
    ).to_list(100)
    
    # Get all expenses in this group within budget period
    expenses = await db.expenses.find({
        "group_id": budget["group_id"],
        "date": {
            "$gte": budget["start_date"],
            "$lte": budget["end_date"]
        }
    }, {"_id": 0}).to_list(10000)
    
    # Calculate total spent
    total_spent = round(sum(exp["amount"] for exp in expenses), 2)
    
    # Calculate per-category spending
    category_spending = {}
    for exp in expenses:
        cat_id = exp.get("category_id")
        if cat_id:
            if cat_id not in category_spending:
                category_spending[cat_id] = 0.0
            category_spending[cat_id] = round(category_spending[cat_id] + exp["amount"], 2)
    
    # Build allocation details with spending info
    allocation_details = []
    for alloc in allocations:
        cat = await db.categories.find_one({"category_id": alloc["category_id"]}, {"_id": 0})
        spent = category_spending.get(alloc["category_id"], 0.0)
        remaining = round(alloc["allocated_amount"] - spent, 2)
        pct_used = round((spent / alloc["allocated_amount"] * 100), 1) if alloc["allocated_amount"] > 0 else 0.0
        
        allocation_details.append({
            "alloc_id": alloc["alloc_id"],
            "category_id": alloc["category_id"],
            "category_name": cat["name"] if cat else "Unknown",
            "category_emoji": cat["emoji"] if cat else "📦",
            "allocated_amount": round(alloc["allocated_amount"], 2),
            "spent": spent,
            "remaining": remaining,
            "percent_used": pct_used,
            "status": "exceeded" if pct_used > 100 else ("warning" if pct_used >= 80 else "ok")
        })
    
    total_allocated = round(sum(a["allocated_amount"] for a in allocations), 2)
    unallocated = round(budget["total_budget"] - total_allocated, 2)
    total_remaining = round(budget["total_budget"] - total_spent, 2)
    total_pct = round((total_spent / budget["total_budget"] * 100), 1) if budget["total_budget"] > 0 else 0.0
    
    return {
        "budget_id": budget["budget_id"],
        "group_id": budget["group_id"],
        "total_budget": round(budget["total_budget"], 2),
        "total_allocated": total_allocated,
        "unallocated": unallocated,
        "total_spent": total_spent,
        "total_remaining": total_remaining,
        "total_percent_used": total_pct,
        "total_status": "exceeded" if total_pct > 100 else ("warning" if total_pct >= 80 else "ok"),
        "start_date": budget["start_date"].isoformat() if isinstance(budget["start_date"], datetime) else budget["start_date"],
        "end_date": budget["end_date"].isoformat() if isinstance(budget["end_date"], datetime) else budget["end_date"],
        "is_recurring": budget.get("is_recurring", False),
        "allocations": allocation_details
    }

@api_router.put("/budgets/{budget_id}")
async def update_budget(budget_id: str, data: GroupBudgetUpdate, current_user: User = Depends(get_current_user)):
    """Update a group budget"""
    budget = await db.group_budgets.find_one({"budget_id": budget_id}, {"_id": 0})
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    group = await db.groups.find_one({"group_id": budget["group_id"]}, {"_id": 0})
    if group["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Only the group creator can edit budgets")
    
    update_fields = {}
    if data.total_budget is not None:
        if data.total_budget <= 0:
            raise HTTPException(status_code=400, detail="Total budget must be positive")
        # Validate against existing allocations
        total_alloc = 0.0
        allocs = await db.category_budgets.find({"budget_id": budget_id}, {"_id": 0}).to_list(100)
        total_alloc = sum(a["allocated_amount"] for a in allocs)
        if data.total_budget < total_alloc:
            raise HTTPException(status_code=400, detail=f"Cannot set budget below allocated amount ({total_alloc})")
        update_fields["total_budget"] = round(data.total_budget, 2)
    
    if data.start_date is not None:
        update_fields["start_date"] = datetime.fromisoformat(data.start_date)
    if data.end_date is not None:
        update_fields["end_date"] = datetime.fromisoformat(data.end_date)
    if data.is_recurring is not None:
        update_fields["is_recurring"] = data.is_recurring
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    await db.group_budgets.update_one({"budget_id": budget_id}, {"$set": update_fields})
    return await db.group_budgets.find_one({"budget_id": budget_id}, {"_id": 0})

@api_router.delete("/budgets/{budget_id}")
async def delete_budget(budget_id: str, current_user: User = Depends(get_current_user)):
    """Delete a group budget and its allocations"""
    budget = await db.group_budgets.find_one({"budget_id": budget_id}, {"_id": 0})
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    group = await db.groups.find_one({"group_id": budget["group_id"]}, {"_id": 0})
    if group["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Only the group creator can delete budgets")
    
    await db.category_budgets.delete_many({"budget_id": budget_id})
    await db.group_budgets.delete_one({"budget_id": budget_id})
    return {"message": "Budget deleted successfully"}

@api_router.post("/budgets/{budget_id}/allocations")
async def add_category_allocation(budget_id: str, data: CategoryBudgetCreate, current_user: User = Depends(get_current_user)):
    """Allocate budget to a category"""
    budget = await db.group_budgets.find_one({"budget_id": budget_id}, {"_id": 0})
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    group = await db.groups.find_one({"group_id": budget["group_id"]}, {"_id": 0})
    if group["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Only the group creator can manage budget allocations")
    
    # Validate category exists
    category = await db.categories.find_one({"category_id": data.category_id}, {"_id": 0})
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    if data.allocated_amount <= 0:
        raise HTTPException(status_code=400, detail="Allocated amount must be positive")
    
    # Check for duplicate
    existing = await db.category_budgets.find_one({
        "budget_id": budget_id,
        "category_id": data.category_id
    })
    if existing:
        raise HTTPException(status_code=400, detail="Category already has an allocation in this budget")
    
    # Validate total allocations don't exceed budget
    current_allocs = await db.category_budgets.find({"budget_id": budget_id}, {"_id": 0}).to_list(100)
    total_allocated = sum(a["allocated_amount"] for a in current_allocs)
    if round(total_allocated + data.allocated_amount, 2) > budget["total_budget"]:
        available = round(budget["total_budget"] - total_allocated, 2)
        raise HTTPException(
            status_code=400,
            detail=f"Allocation exceeds budget. Available: {available}"
        )
    
    alloc_id = f"alloc_{uuid.uuid4().hex[:12]}"
    alloc_doc = {
        "alloc_id": alloc_id,
        "budget_id": budget_id,
        "category_id": data.category_id,
        "allocated_amount": round(data.allocated_amount, 2),
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.category_budgets.insert_one(alloc_doc)
    created = await db.category_budgets.find_one({"alloc_id": alloc_id}, {"_id": 0})
    created["category_name"] = category["name"]
    created["category_emoji"] = category["emoji"]
    return created

@api_router.put("/budgets/{budget_id}/allocations/{alloc_id}")
async def update_category_allocation(
    budget_id: str, alloc_id: str, data: CategoryBudgetUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update a category budget allocation"""
    budget = await db.group_budgets.find_one({"budget_id": budget_id}, {"_id": 0})
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    group = await db.groups.find_one({"group_id": budget["group_id"]}, {"_id": 0})
    if group["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Only the group creator can manage budget allocations")
    
    alloc = await db.category_budgets.find_one({"alloc_id": alloc_id, "budget_id": budget_id})
    if not alloc:
        raise HTTPException(status_code=404, detail="Allocation not found")
    
    if data.allocated_amount <= 0:
        raise HTTPException(status_code=400, detail="Allocated amount must be positive")
    
    # Validate new total doesn't exceed budget
    current_allocs = await db.category_budgets.find({"budget_id": budget_id}, {"_id": 0}).to_list(100)
    total_other = sum(a["allocated_amount"] for a in current_allocs if a["alloc_id"] != alloc_id)
    if round(total_other + data.allocated_amount, 2) > budget["total_budget"]:
        available = round(budget["total_budget"] - total_other, 2)
        raise HTTPException(
            status_code=400,
            detail=f"Allocation exceeds budget. Available: {available}"
        )
    
    await db.category_budgets.update_one(
        {"alloc_id": alloc_id},
        {"$set": {"allocated_amount": round(data.allocated_amount, 2)}}
    )
    return await db.category_budgets.find_one({"alloc_id": alloc_id}, {"_id": 0})

@api_router.delete("/budgets/{budget_id}/allocations/{alloc_id}")
async def delete_category_allocation(
    budget_id: str, alloc_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a category budget allocation"""
    budget = await db.group_budgets.find_one({"budget_id": budget_id}, {"_id": 0})
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    group = await db.groups.find_one({"group_id": budget["group_id"]}, {"_id": 0})
    if group["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Only the group creator can manage budget allocations")
    
    result = await db.category_budgets.delete_one({"alloc_id": alloc_id, "budget_id": budget_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Allocation not found")
    
    return {"message": "Allocation deleted successfully"}

# ============= ANALYTICS ROUTES =============

@api_router.get("/analytics/summary")
async def get_analytics_summary(
    period: str = "month",  # month, week, year
    group_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get expense analytics summary"""
    # Calculate date range
    now = datetime.now(timezone.utc)
    if period == "week":
        start_date = now - timedelta(days=7)
    elif period == "year":
        start_date = now - timedelta(days=365)
    else:  # month
        start_date = now - timedelta(days=30)
    
    # Build query
    query = {
        "user_id": current_user.user_id,
        "date": {"$gte": start_date}
    }
    
    if group_id:
        # Verify membership
        member = await db.group_members.find_one({
            "group_id": group_id,
            "user_id": current_user.user_id
        })
        if not member:
            raise HTTPException(status_code=403, detail="Not a member of this group")
        query = {"group_id": group_id, "date": {"$gte": start_date}}
    else:
        query["group_id"] = None
    
    # Get expenses
    expenses = await db.expenses.find(query, {"_id": 0}).to_list(1000)
    
    # Calculate summary
    total_spend = sum(exp["amount"] for exp in expenses)
    expense_count = len(expenses)
    
    # Category breakdown
    category_breakdown = {}
    for exp in expenses:
        # Handle both new (category_id) and old (category) formats
        cat_name = exp.get("category", "Unknown")
        cat_emoji = exp.get("emoji", "📦")
        
        # If expense uses new category_id format, look up the category
        if "category_id" in exp and exp["category_id"]:
            cat_doc = await db.categories.find_one(
                {"category_id": exp["category_id"]},
                {"_id": 0}
            )
            if cat_doc:
                cat_name = cat_doc["name"]
                cat_emoji = cat_doc["emoji"]
        
        if cat_name not in category_breakdown:
            category_breakdown[cat_name] = {"total": 0, "count": 0, "emoji": cat_emoji}
        category_breakdown[cat_name]["total"] += exp["amount"]
        category_breakdown[cat_name]["count"] += 1
    
    return {
        "total_spend": total_spend,
        "category_breakdown": category_breakdown,
        "expense_count": expense_count,
        "period": period,
        "start_date": start_date.isoformat(),
        "end_date": now.isoformat()
    }

# ============= HEALTH CHECK =============

@api_router.get("/")
async def root():
    return {"message": "Expense Tracker API", "version": "1.0"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

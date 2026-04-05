from fastapi import FastAPI, APIRouter, HTTPException, Depends, Cookie, Response, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import httpx
from emergentintegrations.llm.chat import LlmChat, UserMessage
import json
import base64
import re

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

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

class UserSession(BaseModel):
    user_id: str
    session_token: str
    expires_at: datetime
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
    category_id: str  # Reference to categories collection
    item_name: Optional[str] = None
    date: datetime
    user_id: str
    group_id: Optional[str] = None
    added_by_name: str
    added_by_picture: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime

# Request/Response Models
class SessionCreate(BaseModel):
    session_id: str

class SessionResponse(BaseModel):
    session_token: str
    user: User

class CategoryCreate(BaseModel):
    name: str
    emoji: str
    scope: str  # 'private' or 'group'
    group_id: Optional[str] = None

class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    emoji: Optional[str] = None

class ExpenseParseRequest(BaseModel):
    text: str

class ExpenseParsedResponse(BaseModel):
    amount: float
    category_id: str
    category_name: str
    emoji: str
    item_name: Optional[str] = None
    date: str
    confidence: float
    original_text: str

class ExpenseCreate(BaseModel):
    amount: float
    category_id: str
    item_name: Optional[str] = None
    date: str
    group_id: Optional[str] = None
    notes: Optional[str] = None

class GroupCreate(BaseModel):
    name: str

class GroupUpdate(BaseModel):
    name: str

class GroupInvite(BaseModel):
    user_email: str

class AudioTranscribeRequest(BaseModel):
    audio_base64: str

class AnalyticsSummary(BaseModel):
    total_spend: float
    category_breakdown: dict
    expense_count: int
    period: str

# ============= GLOBAL CATEGORIES =============

GLOBAL_CATEGORIES = [
    {"name": "Food", "emoji": "🍔", "scope": "global"},
    {"name": "Groceries", "emoji": "🛒", "scope": "global"},
    {"name": "Travel", "emoji": "🚕", "scope": "global"},
    {"name": "Rent", "emoji": "🏠", "scope": "global"},
    {"name": "Health", "emoji": "💊", "scope": "global"},
    {"name": "Entertainment", "emoji": "🎮", "scope": "global"},
    {"name": "Miscellaneous", "emoji": "📦", "scope": "global"},
]

async def ensure_global_categories():
    """Ensure global categories exist in database"""
    for cat in GLOBAL_CATEGORIES:
        existing = await db.categories.find_one({"name": cat["name"], "scope": "global"})
        if not existing:
            await db.categories.insert_one({
                "category_id": f"cat_{uuid.uuid4().hex[:12]}",
                "name": cat["name"],
                "emoji": cat["emoji"],
                "scope": cat["scope"],
                "user_id": None,
                "group_id": None,
                "created_by": "system",
                "created_at": datetime.now(timezone.utc)
            })
    logger.info("Global categories initialized")

# ============= CATEGORY VALIDATION =============

async def validate_category_for_expense(category_id: str, group_id: Optional[str], user_id: str):
    """Validate if category can be used for this expense"""
    category = await db.categories.find_one({"category_id": category_id}, {"_id": 0})
    
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    if group_id:
        # Group expense
        if category["scope"] == "private":
            raise HTTPException(
                status_code=400,
                detail="Cannot use private category in group expense"
            )
        if category["scope"] == "group" and category["group_id"] != group_id:
            raise HTTPException(
                status_code=400,
                detail="Category belongs to different group"
            )
    else:
        # Private expense
        if category["scope"] == "group":
            raise HTTPException(
                status_code=400,
                detail="Cannot use group category in private expense"
            )
        if category["scope"] == "private" and category["user_id"] != user_id:
            raise HTTPException(
                status_code=403,
                detail="Cannot use another user's private category"
            )
    
    return category

# ============= AUTH HELPERS =============

async def get_current_user(
    authorization: Optional[str] = None,
    session_token: Optional[str] = Cookie(None),
    request: Request = None
) -> User:
    """Get current authenticated user from session token (cookie or header)"""
    # Try cookie first, then Authorization header
    token = session_token
    if not token and authorization:
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

@api_router.post("/auth/session", response_model=SessionResponse)
async def create_session(data: SessionCreate, response: Response):
    """Exchange session_id from Emergent Auth for session_token"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": data.session_id}
            )
            resp.raise_for_status()
            session_data = resp.json()
        
        # Create or update user
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user_doc = await db.users.find_one({"email": session_data["email"]}, {"_id": 0})
        
        if user_doc:
            user_id = user_doc["user_id"]
            # Update user info
            await db.users.update_one(
                {"user_id": user_id},
                {"$set": {
                    "name": session_data["name"],
                    "picture": session_data.get("picture")
                }}
            )
        else:
            # Create new user
            user_doc = {
                "user_id": user_id,
                "email": session_data["email"],
                "name": session_data["name"],
                "picture": session_data.get("picture"),
                "created_at": datetime.now(timezone.utc)
            }
            await db.users.insert_one(user_doc)
        
        # Create session
        session_token = session_data["session_token"]
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        
        await db.user_sessions.insert_one({
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": expires_at,
            "created_at": datetime.now(timezone.utc)
        })
        
        # Set httpOnly cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=True,
            samesite="none",
            path="/",
            max_age=7*24*60*60
        )
        
        user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
        
        return {
            "session_token": session_token,
            "user": User(**user)
        }
    
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

@api_router.post("/categories", response_model=Category)
async def create_category(data: CategoryCreate, current_user: User = Depends(get_current_user)):
    """Create a custom category (private or group)"""
    
    # Validate scope
    if data.scope not in ["private", "group"]:
        raise HTTPException(status_code=400, detail="Scope must be 'private' or 'group'")
    
    # If group category, verify user is member
    if data.scope == "group":
        if not data.group_id:
            raise HTTPException(status_code=400, detail="group_id required for group category")
        
        member = await db.group_members.find_one({
            "group_id": data.group_id,
            "user_id": current_user.user_id
        })
        if not member:
            raise HTTPException(status_code=403, detail="Not a member of this group")
    
    category_id = f"cat_{uuid.uuid4().hex[:12]}"
    
    category_doc = {
        "category_id": category_id,
        "name": data.name,
        "emoji": data.emoji,
        "scope": data.scope,
        "user_id": current_user.user_id if data.scope == "private" else None,
        "group_id": data.group_id if data.scope == "group" else None,
        "created_by": current_user.user_id,
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.categories.insert_one(category_doc)
    
    return Category(**category_doc)

@api_router.get("/categories", response_model=List[Category])
async def list_categories(
    group_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List available categories based on context"""
    
    query_list = [
        {"scope": "global"},  # Global categories
        {"scope": "private", "user_id": current_user.user_id}  # User's private categories
    ]
    
    # If group context, add group categories
    if group_id:
        # Verify membership
        member = await db.group_members.find_one({
            "group_id": group_id,
            "user_id": current_user.user_id
        })
        if member:
            query_list.append({"scope": "group", "group_id": group_id})
    
    categories = await db.categories.find(
        {"$or": query_list},
        {"_id": 0}
    ).to_list(1000)
    
    return [Category(**cat) for cat in categories]

@api_router.put("/categories/{category_id}", response_model=Category)
async def update_category(
    category_id: str,
    data: CategoryUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update a category (only if owner)"""
    
    category = await db.categories.find_one({"category_id": category_id}, {"_id": 0})
    
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Cannot edit global categories
    if category["scope"] == "global":
        raise HTTPException(status_code=403, detail="Cannot edit global categories")
    
    # Verify ownership
    if category["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this category")
    
    update_data = {}
    if data.name:
        update_data["name"] = data.name
    if data.emoji:
        update_data["emoji"] = data.emoji
    
    if update_data:
        await db.categories.update_one(
            {"category_id": category_id},
            {"$set": update_data}
        )
    
    updated = await db.categories.find_one({"category_id": category_id}, {"_id": 0})
    return Category(**updated)

@api_router.delete("/categories/{category_id}")
async def delete_category(category_id: str, current_user: User = Depends(get_current_user)):
    """Delete a category (only if owner and no expenses use it)"""
    
    category = await db.categories.find_one({"category_id": category_id}, {"_id": 0})
    
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Cannot delete global categories
    if category["scope"] == "global":
        raise HTTPException(status_code=403, detail="Cannot delete global categories")
    
    # Verify ownership
    if category["created_by"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this category")
    
    # Check if any expenses use this category
    expense_count = await db.expenses.count_documents({"category_id": category_id})
    if expense_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete category: {expense_count} expenses use this category"
        )
    
    await db.categories.delete_one({"category_id": category_id})
    
    return {"message": "Category deleted successfully"}

# Rest of the server.py continues in next file...

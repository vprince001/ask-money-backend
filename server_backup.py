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
    date: str
    group_id: Optional[str] = None
    notes: Optional[str] = None

class GroupCreate(BaseModel):
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

# ============= CATEGORY MAPPING =============

CATEGORY_MAP = {
    "food": ("Food", "🍔"),
    "groceries": ("Groceries", "🛒"),
    "grocery": ("Groceries", "🛒"),
    "travel": ("Travel", "🚕"),
    "transport": ("Travel", "🚕"),
    "uber": ("Travel", "🚕"),
    "ola": ("Travel", "🚕"),
    "petrol": ("Travel", "🚕"),
    "fuel": ("Travel", "🚕"),
    "rent": ("Rent", "🏠"),
    "health": ("Health", "💊"),
    "medical": ("Health", "💊"),
    "medicine": ("Health", "💊"),
    "doctor": ("Health", "💊"),
    "entertainment": ("Entertainment", "🎮"),
    "movie": ("Entertainment", "🎮"),
    "game": ("Entertainment", "🎮"),
    "dinner": ("Food", "🍔"),
    "lunch": ("Food", "🍔"),
    "breakfast": ("Food", "🍔"),
    "burger": ("Food", "🍔"),
    "pizza": ("Food", "🍔"),
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

@api_router.post("/auth/session", response_model=SessionResponse)
async def create_session(data: SessionCreate, response: Response):
    """Exchange session_id from Emergent Auth for session_token"""
    # REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
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

# ============= AI PARSING ROUTES =============

@api_router.post("/expenses/parse", response_model=ExpenseParsedResponse)
async def parse_expense(data: ExpenseParseRequest, current_user: User = Depends(get_current_user)):
    """Parse natural language expense using AI"""
    try:
        # Use OpenAI to parse the expense
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"expense_parser_{current_user.user_id}_{uuid.uuid4().hex[:8]}",
            system_message="""You are an expense parsing AI. Extract structured data from natural language expense descriptions.

Response format (JSON only):
{
    "amount": <number>,
    "category": "<one of: food, groceries, travel, rent, health, entertainment, miscellaneous>",
    "date": "<YYYY-MM-DD>",
    "confidence": <0.0-1.0>
}

Rules:
- Amount: Extract numeric value (mandatory)
- Category: Detect from keywords, default to 'miscellaneous' if unclear
- Date: Parse relative dates ('yesterday' = -1 day, 'last sunday' = previous Sunday, 'today' = current date)
- Confidence: 1.0 if all fields clear, 0.5-0.9 if any uncertainty
- Currency: Always assume INR (₹), ignore currency symbols in parsing

Examples:
"I spent 250 rupees on groceries yesterday" → {"amount": 250, "category": "groceries", "date": "2025-07-14", "confidence": 1.0}
"500 for dinner last night" → {"amount": 500, "category": "food", "date": "2025-07-14", "confidence": 0.9}
"Paid 50000 rent today" → {"amount": 50000, "category": "rent", "date": "2025-07-15", "confidence": 1.0}"""
        ).with_model("openai", "gpt-5.2")
        
        user_message = UserMessage(text=f"Parse this expense: '{data.text}'\n\nToday's date is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
        response = await chat.send_message(user_message)
        
        # Parse JSON response
        parsed = json.loads(response)
        
        # Map category to emoji
        category, emoji = get_category_and_emoji(parsed["category"])
        
        return ExpenseParsedResponse(
            amount=parsed["amount"],
            category=category,
            emoji=emoji,
            date=parsed["date"],
            confidence=parsed["confidence"],
            original_text=data.text
        )
    
    except Exception as e:
        logger.error(f"Parse error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse expense: {str(e)}")

# ============= VOICE TRANSCRIPTION =============

@api_router.post("/voice/transcribe")
async def transcribe_audio(data: AudioTranscribeRequest, current_user: User = Depends(get_current_user)):
    """Transcribe audio using Whisper API with OpenAI key"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(data.audio_base64)
        
        # Save temporarily
        temp_path = f"/tmp/audio_{uuid.uuid4().hex[:8]}.m4a"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # Use OpenAI Whisper with the dedicated API key
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            result_text = transcription.text
            logger.info(f"Transcription successful: {result_text}")
        except Exception as openai_error:
            logger.error(f"OpenAI Whisper error: {str(openai_error)}")
            raise HTTPException(status_code=400, detail=f"Whisper API error: {str(openai_error)}")
        
        # Clean up
        os.remove(temp_path)
        
        return {"text": result_text}
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to transcribe audio: {str(e)}")

# ============= EXPENSE ROUTES =============

@api_router.post("/expenses", response_model=Expense)
async def create_expense(data: ExpenseCreate, current_user: User = Depends(get_current_user)):
    """Create a new expense"""
    expense_id = f"expense_{uuid.uuid4().hex[:12]}"
    
    # Parse date
    expense_date = datetime.fromisoformat(data.date) if isinstance(data.date, str) else data.date
    
    # Get category and emoji
    category, emoji = get_category_and_emoji(data.category)
    
    expense_doc = {
        "expense_id": expense_id,
        "amount": data.amount,
        "category": category,
        "emoji": emoji,
        "date": expense_date,
        "user_id": current_user.user_id,
        "group_id": data.group_id,
        "added_by_name": current_user.name,
        "notes": data.notes,
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.expenses.insert_one(expense_doc)
    
    return Expense(**expense_doc)

@api_router.get("/expenses", response_model=List[Expense])
async def list_expenses(
    group_id: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """List expenses (personal or group)"""
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
    else:
        # Personal expenses only (no group)
        query["group_id"] = None
    
    expenses = await db.expenses.find(query, {"_id": 0}).sort("date", -1).limit(limit).to_list(limit)
    
    return [Expense(**exp) for exp in expenses]

@api_router.delete("/expenses/{expense_id}")
async def delete_expense(expense_id: str, current_user: User = Depends(get_current_user)):
    """Delete an expense"""
    # Verify ownership
    expense = await db.expenses.find_one({"expense_id": expense_id, "user_id": current_user.user_id})
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    
    await db.expenses.delete_one({"expense_id": expense_id})
    
    return {"message": "Expense deleted successfully"}

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
        cat = exp["category"]
        if cat not in category_breakdown:
            category_breakdown[cat] = {"total": 0, "count": 0, "emoji": exp["emoji"]}
        category_breakdown[cat]["total"] += exp["amount"]
        category_breakdown[cat]["count"] += 1
    
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

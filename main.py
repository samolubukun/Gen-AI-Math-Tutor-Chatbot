from fastapi import FastAPI, Request, Form, HTTPException, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import google.generativeai as genai
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import base64
from typing import Optional, List
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="AI Math Tutor Pro", description="Enhanced AI Math Tutor with Gemini Flash 1.5")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Google AI Studio Configuration
genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# MongoDB Configuration
MONGODB_URL = "mongodb+srv://samuel:samuelolubukun@cluster0.g8op9yf.mongodb.net/"
client = AsyncIOMotorClient(MONGODB_URL)
db = client.math_tutor_db
users_collection = db.users
chats_collection = db.chats

# Authentication Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await users_collection.find_one({"username": username})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# AI Helper Functions
async def get_math_response(prompt: str, image_data: Optional[bytes] = None) -> str:
    try:
        system_prompt = """You are an expert AI math tutor. Provide detailed, step-by-step solutions with clear explanations. 
        Use LaTeX formatting for mathematical expressions (wrap in $ for inline, $$ for block equations).
        Structure your response with proper HTML formatting using headings, paragraphs, and lists.
        If solving problems, show each step clearly. If explaining concepts, provide examples and visual descriptions."""
        
        if image_data:
            image = Image.open(io.BytesIO(image_data))
            response = model.generate_content([system_prompt + "\n\nAnalyze this math problem in the image and solve it step by step:\n" + prompt, image])
        else:
            response = model.generate_content(system_prompt + "\n\n" + prompt)
        
        return response.text
    except Exception as e:
        return f"Error processing request: {str(e)}"

def generate_graph(expression: str, x_range: tuple = (-10, 10)) -> str:
    """Generate a simple graph for mathematical expressions"""
    try:
        plt.figure(figsize=(8, 6))
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        # Simple expression evaluation (basic functions only for security)
        if 'sin' in expression:
            y = np.sin(x)
            plt.title(f'Graph of sin(x)')
        elif 'cos' in expression:
            y = np.cos(x)
            plt.title(f'Graph of cos(x)')
        elif 'x^2' in expression or 'x**2' in expression:
            y = x**2
            plt.title(f'Graph of x²')
        elif 'x^3' in expression or 'x**3' in expression:
            y = x**3
            plt.title(f'Graph of x³')
        else:
            y = x  # Default to linear
            plt.title(f'Graph of x')
        
        plt.plot(x, y, 'b-', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    # Check if user exists
    existing_user = await users_collection.find_one({"username": username})
    if existing_user:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Username already exists"
        })
    
    # Create new user
    hashed_password = get_password_hash(password)
    user_data = {
        "username": username,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    await users_collection.insert_one(user_data)
    
    # Create access token
    access_token = create_access_token(data={"sub": username})
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie("access_token", access_token, httponly=True)
    return response

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = await users_collection.find_one({"username": username})
    if not user or not verify_password(password, user["password"]):
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Invalid credentials"
        })
    
    access_token = create_access_token(data={"sub": username})
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie("access_token", access_token, httponly=True)
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = await users_collection.find_one({"username": username})
        if not user:
            return RedirectResponse(url="/")
            
        # Get recent chats
        recent_chats = await chats_collection.find(
            {"username": username}
        ).sort("timestamp", -1).limit(10).to_list(length=10)
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": user,
            "recent_chats": recent_chats
        })
    except JWTError:
        return RedirectResponse(url="/")

@app.post("/ask", response_class=JSONResponse)
async def ask_math(
    request: Request,
    question: str = Form(...),
    generate_plot: bool = Form(False),
    image: Optional[UploadFile] = File(None)
):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Process image if provided
    image_data = None
    if image and image.content_type.startswith('image/'):
        image_data = await image.read()
    
    # Get AI response
    result = await get_math_response(question, image_data)
    
    # Generate graph if requested
    graph_data = None
    if generate_plot:
        graph_data = generate_graph(question)
    
    # Save to database
    chat_data = {
        "username": username,
        "question": question,
        "answer": result,
        "graph": graph_data,
        "timestamp": datetime.utcnow(),
        "has_image": image_data is not None
    }
    
    await chats_collection.insert_one(chat_data)
    
    return {
        "question": question,
        "answer": result,
        "graph": graph_data,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/history", response_class=JSONResponse)
async def get_history(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        
        chats = await chats_collection.find(
            {"username": username}
        ).sort("timestamp", -1).limit(50).to_list(length=50)
        
        # Convert ObjectId to string for JSON serialization
        for chat in chats:
            chat["_id"] = str(chat["_id"])
            chat["timestamp"] = chat["timestamp"].isoformat()
        
        return {"chats": chats}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
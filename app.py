from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncAzureOpenAI, OpenAIError
from typing import List
from uuid import uuid4
import base64
import os, openai
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# MongoDB client setup
MONGO_DETAILS = os.getenv("MONGO_DETAILS")
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.chat_app
chat_collection = database.chats

app.mount("/static", StaticFiles(directory="static"), name="static") 
templates = Jinja2Templates(directory="templates")

# Azure OpenAI Client Setup
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION") or "2024-10-01-preview"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or '''You are a helpful assistant.'''



client_ai = AsyncAzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
)

def encode_image(file):
    return base64.b64encode(file).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/create_chat")
async def create_chat():
    session_id = str(uuid4())
    timestamp = datetime.now() 
    await chat_collection.insert_one({
        "session_id": session_id,
        "messages": [],
        "created_at": timestamp
    })
    return {"session_id": session_id}

@app.get("/chats")
async def get_chats():
    sessions = await chat_collection.distinct("session_id")
    return sessions

@app.get("/chat/{session_id}")
async def get_chat(session_id: str):
    chat = await chat_collection.find_one({"session_id": session_id})
    if chat:
        return chat["messages"]
    raise HTTPException(status_code=404, detail="Chat not found")

@app.post("/chat/{session_id}")  
async def chat_view(  
    session_id: str,  
    prompt: str = Form(default=''),  
    images: List[UploadFile] = File(default=[]),  
    systemPrompt: str = Form(default=''),  
    temperature: str = Form(default=''),  
    maxResponseTokens: str = Form(default=''),  
    modelSelection: str = Form(default=''),  
    prevMessageCount: int = Form(default=0)  # Parameter to specify number of previous messages  
):  
    # Log the received previous message count  
    #print(f"Previous Message Context Count: {prevMessageCount}")  
  
    chat = await chat_collection.find_one({"session_id": session_id})  
    if not chat:  
        raise HTTPException(status_code=404, detail="Chat not found")  
  
    prompt = prompt.strip()  
    system_prompt = systemPrompt or SYSTEM_PROMPT  
    max_tokens = int(maxResponseTokens) if maxResponseTokens else MAX_TOKENS  
    temp = float(temperature) if temperature else 0.7  
    model = modelSelection or DEPLOYMENT_NAME  
  
    # For normal mode  
    message_entry = {"role": "user", "content": []}  
    if prompt:  
        message_entry["content"].append({"type": "text", "text": prompt})  
    for file in images:  
        base64_image = encode_image(await file.read())  
        message_entry["content"].append({  
            "type": "image_url",  
            "image_url": {  
                "url": f"data:image/jpeg;base64,{base64_image}",  
            },  
        })  
  
    # Save the user's message to the database  
    chat["messages"].append(message_entry)  
    await chat_collection.update_one({"session_id": session_id}, {"$set": {"messages": chat["messages"]}})  
  
    # Use only the last `prevMessageCount` messages for context  
    if prevMessageCount > 0:  
        limited_messages = chat["messages"][-prevMessageCount:]  
    else:  
        limited_messages = chat["messages"]  
  
    # Construct the messages for the AI, including the system prompt  
    all_messages = [{"role": "system", "content": system_prompt}] + limited_messages  
  
    # Define the async response streaming function  
    async def stream_response():  
        try:  
            completion = await client_ai.chat.completions.create(  
                model=model,  
                messages=all_messages,  
                max_tokens=max_tokens,  
                temperature=temp,  
                top_p=0.95,  
                stream=True  
            )  
            async for chunk in completion:  
                if chunk.choices and chunk.choices[0].delta.content:  
                    part = chunk.choices[0].delta.content  
                    yield json.dumps({  
                        "role": "assistant",  
                        "content": part  
                    }, ensure_ascii=False) + "\n"  
        except OpenAIError as e:
            print("body:",e.body)
            # Check if the error is a content policy violation
            if e.body['code']=='content_filter' and e.body['innererror']['code']=='ResponsibleAIPolicyViolation':
                yield json.dumps({
                    "role": "assistant",
                    "content": "This message violates content management policy. Please modify your prompt and retry."
                }, ensure_ascii=False) + "\n"
            else:
                yield json.dumps({
                    "role": "assistant",
                    "content": "Error occured."
                }, ensure_ascii=False) + "\n"

        except openai.RateLimitError as e:
            yield json.dumps({
                "role": "assistant",
                "content": "Rate limit exceeded."
            }, ensure_ascii=False) + "\n" 
  
    # Return the streaming response  
    return StreamingResponse(stream_response(), media_type='application/json')  
    
@app.post("/save_response/{session_id}")
async def save_response(session_id: str, ai_response: str = Form(...)):
    chat = await chat_collection.find_one({"session_id": session_id})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save the full AI response to the database
    chat["messages"].append({"role": "assistant", "content": ai_response})
    await chat_collection.update_one({"session_id": session_id}, {"$set": {"messages": chat["messages"]}})
    return {"status": "success"}

@app.get("/chat_history")
async def chat_history():
    sessions = await chat_collection.find({}, {"session_id": 1, "messages": 1, "created_at": 1}).to_list(length=None)
    history = []
    for session in sessions:
        first_message = ""
        if session["messages"]:
            # Find the first user message
            for message in session["messages"]:
                if message["role"] == "user" and message["content"]:
                    try:
                        first_message = message["content"][0]["text"]
                    except:
                        first_message = message["content"]
                    break
        
        if len(first_message) <= 35:
            history.append({
                "session_id": session["session_id"],
                "summary": f"{first_message} \n {session['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
            })
        else:
            history.append({
                "session_id": session["session_id"],
                "summary": f"{first_message}... \n {session['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
            })
    
    # Reverse the history list
    history.reverse()
    
    return history

@app.delete("/chat/{session_id}")  
async def delete_chat(session_id: str):  
    result = await chat_collection.delete_one({"session_id": session_id})  
    if result.deleted_count == 1:  
        return {"status": "success"}  
    raise HTTPException(status_code=404, detail="Chat not found")  
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncAzureOpenAI, OpenAIError
import openai
from typing import List
from uuid import uuid4
import base64
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

app = FastAPI()

# MongoDB client setup
MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.chat_app
chat_collection = database.chats

app.mount("/static", StaticFiles(directory="static"), name="static") 
templates = Jinja2Templates(directory="templates")

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME") or 'gpt-4o-mini'
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or 'You are a helpful assistant.'


SEARCH_ENDPOINT = os.getenv('SEARCH_ENDPOINT')
SEARCH_KEY =  os.getenv('SEARCH_KEY')
SEARCH_INDEX  = os.getenv('SEARCH_INDEX')

token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

client_ai = AsyncAzureOpenAI(
    api_version='2024-05-01-preview',
    azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider
)

def encode_image(file):
    return base64.b64encode(file).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/create_chat")
async def create_chat():
    session_id = str(uuid4())
    timestamp = datetime.now() + timedelta(hours=5, minutes=30)
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
    ragMode: bool = Form(default=False)
):
    #print("Received Settings:")
    #print(f"System Prompt: {systemPrompt}")
    #print(f"Temperature: {temperature}")
    #print(f"Max Response Tokens: {maxResponseTokens}")
    #print(f"Model Selection: {modelSelection}")
    #print(f"RAG Mode: {ragMode}")  # Print RAG mode status
    chat = await chat_collection.find_one({"session_id": session_id})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    prompt = prompt.strip()
    system_prompt = systemPrompt or SYSTEM_PROMPT
    max_tokens = int(maxResponseTokens) if maxResponseTokens else MAX_TOKENS
    temp = float(temperature) if temperature else 0.7
    model = modelSelection or DEPLOYMENT_NAME

    
    #for RAG mode
    if ragMode:
        if prompt:
            message_entry = {"role": "user", "content": prompt}
            chat["messages"].append(message_entry)
            await chat_collection.update_one({"session_id": session_id}, {"$set": {"messages": chat["messages"]}})
            all_messages = [{"role": "system", "content": system_prompt}] + chat["messages"]
    
    #for normal mode
    else:
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

        all_messages = [{"role": "system", "content": system_prompt}] + chat["messages"]

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
                if chunk.choices and chunk.choices[0].delta:
                    part = chunk.choices[0].delta.content or ''
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

    async def stream_response_RAG():
        try:
            completion =  await client_ai.chat.completions.create(
                model=model,  
                messages=all_messages,  
                max_tokens=max_tokens,  
                temperature=temp,  
                top_p=0.95,   
                stream=True,
                extra_body={
                "data_sources": [{
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": f"{SEARCH_ENDPOINT}",
                        "index_name": f"{SEARCH_INDEX}",
                        "semantic_configuration": f"{SEARCH_INDEX}-semantic-configuration",
                        "query_type": "vector_semantic_hybrid",
                        "fields_mapping": {},
                        "in_scope": True,
                        "role_information": "You are an AI assistant that helps people find information.",
                        "filter": None,
                        "strictness": 3,
                        "top_n_documents": 5,
                        "authentication": {
                        "type": "api_key",
                        "key": f"{SEARCH_KEY}"
                        },
                        "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": "text-embedding-ada-002"
                        }
                    }
                    }]
                }
            )
            async for chunk in completion:
                if chunk.choices and chunk.choices[0].delta:
                    part = chunk.choices[0].delta.content or ''
                    yield json.dumps({
                        "role": "assistant",
                        "content": part
                    }, ensure_ascii=False) + "\n"
        except Exception as e:   
            yield json.dumps({  
                "role": "assistant",  
                "content": "This message violates our policies."  + str(e)
            }, ensure_ascii=False) + "\n"  

    if ragMode:
        return StreamingResponse(stream_response_RAG(), media_type='application/json')
    else:
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
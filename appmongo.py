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
import os, time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

# Initialize Chroma client
client = chromadb.Client(Settings(anonymized_telemetry=False))

# Create a collection
collection = client.create_collection(name="test_collection")

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
    azure_endpoint='https://just-testing-12.openai.azure.com',
    azure_ad_token_provider=token_provider
)

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            ".",
            "\n",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
        ],
        # Set a really small chunk size, just to show.
        chunk_size=1536,
        chunk_overlap=128,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([text])
    #print("chunked text", len(texts), type(texts))
    #print(texts, len(texts))
    return texts

def search(query, max_results=4):
    """Perform a search using DuckDuckGo Search API."""
    results = DDGS().text(query, max_results=max_results)
    return [result['href'] for result in results]

async def clean_markdown_links(markdown_text):
    # Remove image links completely
    markdown_text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)
    # Remove empty markdown links: [](...)
    markdown_text = re.sub(r'\[\]\(.*?\)', '', markdown_text)
    # Replace normal markdown links with just the text
    markdown_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown_text)
    return markdown_text

async def crawl(urls):
    md_list = []
    browser_config = BrowserConfig(
        headless=True,  
        verbose=True
    )
    options = {
        "incognito": True,  # Enable incognito mode
        "disable_cache": True,  # Disable cache
    }
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.DISABLED,  # Disable cache
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=.4, threshold_type="fixed", min_word_threshold=0)
        ),
    )
    
    async with AsyncWebCrawler(config=browser_config, options=options) as crawler:
        # Specify an incognito-friendly URL or settings if supported elsewhere
        results = await crawler.arun_many(urls, config=run_config)
        for result in results:  # Now you can iterate normally
            if result.success:
                __temp = await (clean_markdown_links(result.markdown))
                md_list.append(__temp)
    
    return md_list

async def get_embeddings(inputs):
    response = await client_ai.embeddings.create(
        input=inputs,
        model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings

def encode_image(file):
    return base64.b64encode(file).decode('utf-8')

def extract_text_from_markdown(markdown_content):
    # Replace markdown links [text](url) with just text
    text_only = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown_content)
    return text_only

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

        #print(all_messages)
        if all_messages[0] == {'role': 'system', 'content': 'You are a helpful assistant.'}:
            __temp_comtext = all_messages[1:]
        else:
            __temp_comtext = all_messages

        #context aware prompting
        __temp_prompt = await client_ai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "you are a search assistant/ prompt maker , you will be given a question and context try to make prompt for question\
                     DO NOT ANSWER THE QUESTION, just make simple short prompt for question and context, make sure not to use long sentences, as searching long sentence will fetch poor results,\
                     DO not add year or date in prompt unless specified"},
                ] + __temp_comtext,
                max_tokens=30,
                temperature=0.0,
                top_p=0.0
        )       
        
        context_aware_prompt = __temp_prompt.choices[0].message.content
        print("context aware prompt", context_aware_prompt)


        # Define the functions schema
        functions = [
            {
                "name": "summarise_website",
                "description": "summarise, extract text, describe from a URL and will provide the URL. DO not modify prompt or URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "summarise https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling"
                        },
                        "URL": {
                            "type": "string",
                            "description": "https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling"
                        }
                    },
                    "required": ["prompt", "URL"]
                }
            },
            {
                "name": "web_search",
                "description": "Perform a web search for a query if specific knowledge is not available in training data and query requires real-time data. DO not modify prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Tell me the latest news"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        ]

        response = await client_ai.chat.completions.create(  # Add 'await' here
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": context_aware_prompt}
            ],
            functions=functions,
            function_call="auto"
        )

        # Handle the function call
        message = response.choices[0].message  # No need to await here anymore


    async def summarise_website(prompt, URL):
        if URL.endswith('.jpg') or URL.endswith('.png') or URL.endswith('.jpeg'):
            completion = await client_ai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": URL}},
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=temp,
                top_p=0.95,
                stream=True,  # Ensure streaming is enabled
            )

            # Handle the streaming response
            async for chunk in completion:
                if chunk.choices and chunk.choices[0].delta:
                    part = chunk.choices[0].delta.content or ''
                    yield json.dumps({
                        "role": "assistant",
                        "content": part
                    }, ensure_ascii=False) + "\n"

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
    
    async def web_search(prompt):
        print("in web search")
        query = prompt
        urls = search(query)
        md_list = await crawl(urls)
        to_upsert = []
        all_texts = []
        all_chunks = []
        a=0
        for j in md_list:
            # Generate embeddings for text chunks
            __temp_chunk = chunk_text(j)
            if len(__temp_chunk) >8:
                __temp_chunk = __temp_chunk[:8]  # Limit to 8 chunks
            all_chunks.extend(__temp_chunk)  # Collect all chunks for embedding
            website_texts = [str(chunk.page_content) for chunk in __temp_chunk]
            all_texts.extend(website_texts)  # Collect all texts for embedding
        

        embeddings = await get_embeddings(all_texts)
        collection.add(
            documents=all_texts,
            embeddings=embeddings,
            ids=[f"doc{i}" for i in range(len(all_texts))]
        )

        query_embedding = await get_embeddings([query])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=4,
            include=["documents", "distances"]
        )

        similar_context = ""
        for doc in (results['documents'][0]):
            similar_context += f"Document: {doc}\n\n"

        completion = await client_ai.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "system", 
                    "content": 'You are a helpful assistant. You will be given a question and a context try to answer question.\n---------context-------\n.'+"\n\n" + similar_context+"\n\n" + "------------------\n\n" + "Along with the answer put these links as citations in response \n"+ str(urls)
                },
                {
                    "role": "user",
                    "content": f"Answer the question: {query}"
                }
            ],
            max_tokens=1536,
            temperature=0.7,
            top_p=0.95,
            stream=True
        )
        async for chunk in completion:
            if chunk.choices and chunk.choices[0].delta:
                if chunk.choices and chunk.choices[0].delta:
                    part = chunk.choices[0].delta.content or ''
                    yield json.dumps({
                        "role": "assistant",
                        "content": part
                    }, ensure_ascii=False) + "\n"


    if ragMode:
        return StreamingResponse(stream_response_RAG(), media_type='application/json')
    else:
        #return StreamingResponse(web_search(prompt), media_type='application/json')
        # Handle the function call
        message = response.choices[0].message

        if message.function_call is not None:
            function_name = message.function_call.name
            arguments = json.loads(message.function_call.arguments)
            print("args",arguments['prompt'])
            
            if function_name == "summarise_website":
                print("summarising ....")
                return StreamingResponse(summarise_website(arguments['prompt'], arguments['URL']), media_type='application/json')
            elif function_name == "web_search":
                print("web searching ....")
                return StreamingResponse(web_search(arguments['prompt']), media_type='application/json')
            else:
                print("Unknown function.")
                return StreamingResponse("Unknown function.", media_type='application/json')
        else:
            print("No function call.")
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
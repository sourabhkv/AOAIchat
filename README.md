# AOAIchat
AOAI chat
Open source chat app using Azure OpenAI, Fastapi, CosmosDB.

## Features
- Private GPT (powered by AOAI / GitHub models / Ollama / LM studio)
- Multimodal (text, images) saved in DB
- Scalable
- Secured by [Microsoft Entra](https://learn.microsoft.com/en-us/azure/app-service/configure-authentication-provider-aad?tabs=workforce-configuration&WT.mc_id=studentamb_264449) (Requires Azure deployment)
- Playground env in chat
- Supports RAG on Azure blob storage, Azure SQL, Onelake.


## Screenshots
![Screenshot 2024-12-15 160409](https://github.com/user-attachments/assets/3b54740a-aadd-4199-bf0b-019288bc11fb)
![Screenshot 2024-12-15 160423](https://github.com/user-attachments/assets/3a712401-ab90-4e34-b7a0-75a2c69d6c58)
![Screenshot 2024-12-15 160454](https://github.com/user-attachments/assets/df19ad86-c525-4f93-b992-0bd1fd3523cf)
![Screenshot 2024-12-15 160531](https://github.com/user-attachments/assets/bb3be3db-231b-4c1e-bc2f-543d5724ed0d)




![image](https://github.com/user-attachments/assets/98d9175b-54be-48a2-a8ff-66a296897bfb)

## FAQ
- Can I use Llama / Mistral /Phi models?
  - Yes. [Refer this](https://github.com/microsoft/Phi-3CookBook/blob/main/md/02.QuickStart/OpenAISDK_Quickstart.md)

- Can I run it locally?
  - Yes.
  - Cosmos DB <-> MongoDB. Change  `MONGO_DETAILS = 'mongodb://localhost:27017/'` in `.env`
  - Azure OpenAI <-> [Ollama](https://ollama.com/) / [LM studio](https://lmstudio.ai/docs/api/server) [Refer this](https://github.com/microsoft/Phi-3CookBook/blob/main/md/02.QuickStart/OpenAISDK_Quickstart.md)<br>
    *NOTE : I have used `gpt-4o`, `gpt-4o-mini` these values are hardcoded in webpage, if you are using other models, you might have to change them in `index.html`*.
  - App Service <-> Local machine

- Does it support RAG?
  - Yes, currently it supports RAG only with Azure AI search connected to blob storage, Azure SQL, Onelake.
  - Replace this with your `client`.

    ```python
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
                        "role_information": "",
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
    ```

- Why does client send AI message?
  - The response from Azure OpenAI is getting streamed in realtime, client acts like a buffer to accumulate full response, then sends full AI message to backend to store in database.

- How much does it cost?
  - Pricing may vary if you deploy on Azure and region and configuration. See [Azure calculator](https://azure.microsoft.com/en-in/pricing/calculator?WT.mc_id=studentamb_264449).
  - It can also be **FREE** if you run entirely on your local machine.
    
- Can I use OpenAI credits instead of Azure OpenAI?
  - Yes
 


## Getting Started

### Running with Docker

#### Pull the Docker Image

Pull the Docker image from Docker Hub using the following command:

```bash
docker pull sourabhkv/aoaichatdb:0.1
```

#### Environment Variables

The application requires the following environment variables to be set for proper configuration:

| Environment Variable    | Description                         |
| ----------------------- | ----------------------------------- |
| `AZURE_OPENAI_ENDPOINT` | The endpoint for Azure OpenAI API.  |
| `AZURE_OPENAI_API_KEY`  | API key for accessing Azure OpenAI. |
| `DEPLOYMENT_NAME`       | Azure OpenAI deployment name.       |
| `API_VERSION`           | API version for Azure OpenAI.       |
| `MAX_TOKENS`            | Maximum tokens for API responses.   |
| `MONGO_DETAILS`         | MongoDB connection string.          |

#### Running the Application with Docker

1. Create a `.env` file in your working directory and populate it with the required environment variables. For example:

   ```env
   AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
   AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>
   DEPLOYMENT_NAME=<your_deployment_name>
   API_VERSION=<your_api_version>
   MAX_TOKENS=<max_tokens>
   MONGO_DETAILS=<your_mongo_connection_string>
   ```

2. Run the Docker container, mapping port 80 of the container to port 8000 of your system:

   ```bash
   docker run -d --name aoaichatdb -p 8000:80 --env-file .env sourabhkv/aoaichatdb:0.1
   ```

#### Accessing the Application

Once the container is running, you can access the application at:

```
http://localhost:8000
```

### Running Without Docker

#### Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository_url>
cd <repository_name>
```

#### Set Up Environment Variables

Create a `.env` file in the project root directory and populate it with the required environment variables. For example:

```env
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>
DEPLOYMENT_NAME=<your_deployment_name>
API_VERSION=<your_api_version>
MAX_TOKENS=<max_tokens>
MONGO_DETAILS=<your_mongo_connection_string>
```

#### Create a Virtual Environment and Install Dependencies

1. Create a Python virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

#### Run the Application

Start the FastAPI application using Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### Accessing the Application

Once the application is running, you can access it at:

```
http://localhost:8000
```

### Development and Debugging

If you want to run the application with live reloading for development purposes, use the following command:

```bash
uvicorn app:app --reload
```

## Notes

- Ensure all environment variables are correctly set to avoid configuration issues.
- MongoDB / Cosmos DB and Azure OpenAI services should be correctly configured and accessible.

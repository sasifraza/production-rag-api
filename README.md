# Production RAG API

This project implements a production-style Retrieval-Augmented Generation (RAG) pipeline using LangChain, FAISS, OpenAI, and FastAPI.

## Features

- Document ingestion from TXT files  
- Text chunking for retrieval  
- Vector embeddings using HuggingFace embeddings  
- Semantic retrieval with FAISS  
- Grounded answer generation using an LLM  
- FastAPI endpoint for question answering  

## Project Structure

.
├── app  
│   └── main.py  
├── docs  
│   └── abstract_only.txt  
├── README.md  
├── requirements.txt  
└── src  
&nbsp;&nbsp;&nbsp;&nbsp;├── __init__.py  
&nbsp;&nbsp;&nbsp;&nbsp;├── generate.py  
&nbsp;&nbsp;&nbsp;&nbsp;├── ingest.py  
&nbsp;&nbsp;&nbsp;&nbsp;└── retrieve.py  

## How It Works

The pipeline follows four steps:

1. Ingest the source document  
2. Chunk the text into smaller pieces  
3. Embed the chunks and store them in FAISS  
4. Retrieve relevant chunks and generate a grounded answer with an LLM  

## Tech Stack

- Python  
- LangChain  
- FAISS  
- FastAPI  
- OpenAI  
- HuggingFace Embeddings  

## Setup

Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Create a .env file in the project root:

OPENAI_API_KEY=your_api_key_here

Run the Pipeline

Build the vector store:

python src/ingest.py

Test generation:

python -m src.generate

Run the API:

uvicorn app.main:app --reload

API Usage

http://127.0.0.1:8000/docs

Use /ask with:

{
  "query": "What is the objective of this study?"
}
Notes
	•	.env, venv/, vectorstore/, and __pycache__/ should not be committed
	•	Retrieval quality depends on document quality and chunking

Future Improvements
	•	Add PDF ingestion with better preprocessing
	•	Add reranking
	•	Add Streamlit UI
	•	Add evaluation
	•	Deploy to cloud
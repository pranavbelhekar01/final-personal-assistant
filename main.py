from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
import requests
import json
# from mangum import Mangum
import fitz  # PyMuPDF
from langchain_community.retrievers import ArxivRetriever
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_stack import get_summary, get_gemini_resonse

load_dotenv()

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Specify the path to the data folder containing PDF files
data_folder = "Data"

def google_search(query):
    # Replace 'YOUR_API_KEY' and 'YOUR_CX' with your API key and Custom Search Engine ID (cx)
    api_key = os.getenv('GOOGLE_API_KEY')
    cx = os.getenv('GOOGLE_CX')

    base_url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        'key': api_key,
        'cx': cx,
        'q': query,
    }

    response = requests.get(base_url, params=params)
    results = response.json()

    # Extract top 3 search results (title, snippet/description, link, and image link if provided)
    data = []
    for n in range(min(3, len(results.get('items', [])))):
        title = results['items'][n].get('title', 'N/A')
        snippet = results['items'][n].get('snippet', 'N/A')
        link = results['items'][n].get('link', 'N/A')
        image_link = results['items'][n].get('pagemap', {}).get('cse_image', [{}])[0].get('src', 'N/A')

        data.append({
            'title': title,
            'snippet': snippet,
            'link': link,
            'image_link': image_link,
        })

    return data



def get_relevant_document(query):
    retriever = ArxivRetriever()
    docs = retriever.get_relevant_documents(query=query, max_docs=3)

    if not docs:
        return {
            'Title': 'No paper found on this topic',
            'Link': 'No link',
            'Content': 'No full paper available...'
        }

    contents = [doc.page_content for doc in docs]
    vectorizer = TfidfVectorizer()
    tfidfs = vectorizer.fit_transform(contents)

    query_tfidf = vectorizer.transform([query])
    similarities = tfidfs * query_tfidf.T
    most_similar_idx = similarities.argmax()
    most_similar_doc = docs[most_similar_idx]

    title = most_similar_doc.metadata['Title']
    entry_id = most_similar_doc.metadata['Entry ID']

    # Extract the ArXiv ID from the Entry ID (if it contains the full URL)
    if entry_id.startswith("http"):
        entry_id = entry_id.split('/')[-1]

    pdf_link = f"https://arxiv.org/pdf/{entry_id}.pdf"

    try:
        # Download the PDF
        response = requests.get(pdf_link)
        response.raise_for_status()

        # Save the PDF temporarily
        pdf_path = f"Data/temp.pdf"
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)

        # Extract text using PyMuPDF
        full_text = extract_full_text_from_pdf(pdf_path)

        summary = get_summary(full_text)  

    except Exception as e:
        full_text = "Failed to retrieve the full paper. Error: " + str(e)

    return {
        'Title': title,
        'Link': pdf_link,
        'Content': summary
    }

def extract_full_text_from_pdf(pdf_path):
    try:
        full_text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text()
        return full_text
    except Exception as e:
        return f"Error during text extraction: {str(e)}"
    

def youtube_search(query):
    # Replace 'YOUR_YOUTUBE_API_KEY' with your YouTube Data API key
    api_key = os.getenv('YOUTUBE_API_KEY')

    base_url = "https://www.googleapis.com/youtube/v3/search"
    
    params = {
        'key': api_key,
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': 3,
    }

    response = requests.get(base_url, params=params)
    results = response.json()

    # Extract top 3 YouTube video results (title, description, and video link)
    data = []
    for item in results.get('items', []):
        title = item['snippet'].get('title', 'N/A')
        description = item['snippet'].get('description', 'N/A')
        video_link = f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'

        data.append({
            'title': title,
            'description': description,
            'video_link': video_link,
        })

    return data



def user_input(user_question):
    gemini_response = get_gemini_resonse(user_question)

    

    # Get response from Google Search API
    google_search_response = google_search(user_question)

    # Get response from YouTube API
    youtube_response = youtube_search(user_question)

    # research papers
    arxiv_response = get_relevant_document(user_question)
    

    return {
        "gemini_response": gemini_response,
        "google_search_response": google_search_response,
        "youtube_response": youtube_response,
        "research_papers": arxiv_response
    }


class QuestionRequest(BaseModel):
    question: str

# Define the /ask endpoint
@app.post('/ask')
async def ask_question(request: QuestionRequest):
    try:
        user_question = request.question

        if not user_question.strip():  # Ensure the question is not empty or just whitespace
            raise HTTPException(status_code=400, detail="Empty question")

        # Process the user question (replace this with your actual implementation)
        responses = user_input(user_question)

        return JSONResponse(content={"responses": responses})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# # Add this line
# handler = Mangum(app)

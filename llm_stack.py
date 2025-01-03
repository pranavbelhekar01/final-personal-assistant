
import google.generativeai as genai
from google.generativeai import caching
import datetime
import time
from templates import prompt_template, gemini_prompt
import fitz 
import os
from google import genai
# from google.genai import types
from dotenv import load_dotenv

load_dotenv()



API_KEY = os.getenv("GOOGLE_GEN_API_KEY")
client = genai.Client(api_key=API_KEY)

def get_summary(content):

    response = client.models.generate_content(
        model='gemini-1.5-flash-002', contents=[content, prompt_template]
    )
    return response.text

def get_gemini_resonse(question):

    response = client.models.generate_content(
        model='gemini-1.5-flash-002', contents=[question, gemini_prompt]
    )
    return response.text

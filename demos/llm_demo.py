from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

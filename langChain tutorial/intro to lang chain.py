from langchain_openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    
    openai_api_key=openai_api_key
    
    
)

template = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""")

langchain = template | llm

response = langchain.invoke({"fruit": "apple"})

print(response)
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader

# pip install --upgrade --quiet duckduckgo-search
# pip install -U duckduckgo_search==5.3.1b1
# the above solved rate limit exception

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0
    )




directory = '/Users/thuptenwangpo/Documents/GitHub/neo4j-practice/Sample_PDFs'
file_paths = []
docs = [PyMuPDFLoader(file_path).load() for file_path in file_paths]
docs_string = "".join(doc.page_content for doc in docs)

template = PromptTemplate.from_template("""
You are a helpful assistant who is good at analyzing source information and answering questions.       Use the following source documents to answer the user's questions.       If you don't know the answer, just say that you don't know.       Use three sentences maximum and keep the answer concise.
Documents:
{docs_string}
User question: 
{question}
Answer:
""".strip())



langchain2 = template | llm


question = input("Enter a question: ")
response = langchain2.invoke({"question": question, "docs_string": docs_string})

print(response.content)
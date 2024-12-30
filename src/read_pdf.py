from pypdf import PdfReader
import spacy
import spacy.cli
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

import pymupdf4llm



#function to get text from pdf
def read_pdf(pdf_name):
    return pymupdf4llm.to_markdown(pdf_name)
directory = '/Users/thuptenwangpo/Documents/GitHub/neo4j-practice/Sample_PDFs'
pdf_files = []

for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        pdf_files.append(os.path.join(directory, filename))

text = ""
for file in pdf_files:
    text = text+read_pdf(file)

# print(text)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=openai_api_key,
    temperature=0
)
llm_transformer = LLMGraphTransformer(llm=llm)


load_dotenv()
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("Neo4j_PASSWORD2")
graph=Neo4jGraph(url=NEO4J_URI,
                 username=NEO4J_USERNAME,
                 password=NEO4J_PASSWORD)

documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)
for node in graph_documents[0].nodes:
    print(node)
for relationship in graph_documents[0].relationships:
    print(relationship)

graph.add_graph_documents(graph_documents)
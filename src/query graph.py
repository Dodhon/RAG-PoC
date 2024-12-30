from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
graphPassword = os.getenv("Neo4j_PASSWORD2")

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=openai_api_key
)
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password=graphPassword,
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema. Only use entities and relationships present in the schema. Don't use any ">" or "<". Don't use directed relationships.

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    allow_dangerous_requests=True,
    # return_direct=True, 
    # return_direct seems to be overridden by return_intermediate_steps
    return_intermediate_steps=False,
    verbose=True
    
    #cypher_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    #qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    # We can have separate LMs for each step
)
try:
    query = "whats is pauls id"
    result = cypher_chain.invoke({"query": query})
except:
    pass



#print(f"result is {result}")
#print(f"Intermediate steps: {result["intermediate_steps"]}")


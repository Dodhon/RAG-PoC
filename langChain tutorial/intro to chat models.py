from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.tools import YouTubeSearchTool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from uuid import uuid4
import os
from dotenv import load_dotenv

load_dotenv()

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key)

password = os.loadenv("Neo4j_PASSWORD2")
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="pleaseletmein"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie expert. You find movies from a genre or plot.",
        ),
        ("human", "{input}"),
    ]
)

movie_chat = prompt | llm | StrOutputParser()

youtube = YouTubeSearchTool()

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

def call_trailer_search(input):
    input = input.replace(",", " ")
    return youtube.run(input)

tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=movie_chat.invoke,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
        func=call_trailer_search,
    ),
]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
    q = input("> ")

    response = chat_agent.invoke(
        {
            "input": q
        },
        {"configurable": {"session_id": SESSION_ID}},
    )
    
    print(response["output"])
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# pip install --upgrade --quiet duckduckgo-search
# pip install -U duckduckgo_search==5.3.1b1
# the above solved rate limit exception

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key
    )

duckduckgo = DDGS(timeout=20)

question = input("Enter a question: ")


template1 = PromptTemplate.from_template("""You are an analyst tasked with converting questions to a form more suitable for search engine queries. Convert the following to a search engine query: {question}. Query: 
""".strip())

langchain1 = template1 | llm
#######
search_query = langchain1.invoke({"question": question})


print("-- LLM Generated search query:", search_query.content)

search_result = duckduckgo.text(search_query.content, max_results=7)
search_results = "\n\n".join(
    [
        "{_index}. {_title}\n{_body}\nSource: {_href}".format(
            _index=i + 1,
            _title=result["title"],
            _body=result["body"],
            _href=result["href"],
        )
        for i, result in enumerate(search_result)
    ]
)

template2 = PromptTemplate.from_template("""
You are a helpful search assistant. Be nice, talkative and informative.
Answer the user's question based on the search results below:
Search results:
{search_results}
User question: 
{question}
Answer:
""".strip())





langchain2 = template2 | llm

response = langchain2.invoke({"question": question, "search_results": search_results})

print(response.content)
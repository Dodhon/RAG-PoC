from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict

# pip install --upgrade --quiet duckduckgo-search
# pip install -U duckduckgo_search==5.3.1b1
# the above solved rate limit exception
load_dotenv()
os.environ['USER_AGENT'] = 'USER_AGENT'
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0
    )




directory = '/Users/thuptenwangpo/Documents/GitHub/neo4j-practice/Sample_PDFs'
file_paths = []
docs = [PyMuPDFLoader(file_path).load() for file_path in file_paths]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)
docs_string = "".join(doc.page_content for doc in docs)

# Add the document chunks to the "vector store" using OpenAIEmbeddings
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever(k=6)

print("")
question = input("Please note that this is AI generated content. What is your question: ")

docs = retriever.invoke(question)

docs_string = "".join(doc.page_content for doc in docs)

template = PromptTemplate.from_template("""
You are a helpful assistant who is good at analyzing source information and answering questions.       
Use the following source documents to answer the user's questions.      
If you don't know the answer, just say  'I don't know'.       
If the answer is not directly in the documents, just say 'I don't know'.
Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}
User question: 
{question}
Answer:
""".strip())

langchain2 = template | llm



response = langchain2.invoke({"question": question, "docs_string": docs_string})
output = response.content
if output == "I don't know.":
    print("I don't know. I will now look online. Please note that the following information is ai generated.")
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from duckduckgo_search import DDGS
    import os
    from dotenv import load_dotenv

    # pip install --upgrade --quiet duckduckgo-search
    # pip install -U duckduckgo_search==5.3.1b1
    # the above solved rate limit exception

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=openai_api_key,
        temperature=0
        )

    duckduckgo = DDGS(timeout=20)



    template1 = PromptTemplate.from_template("""You are an analyst tasked with converting questions to a form more suitable for search engine queries. Convert the following to a search engine query: {question}. Query: 
    """.strip())

    langchain1 = template1 | llm
    #######
    search_query = langchain1.invoke({"question": question})


    #print("-- LLM Generated search query:", search_query.content)

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
    You are a helpful search assistant. Be polite and informative. Always try to not be wrong.
    Only use the search results below to answer. Answer the user's question based on the search results below.
    Use three sentences maximum and keep the answer concise.
    
    Search results:
    {search_results}
    User question: 
    {question}
    Answer:
    """.strip())





    langchain2 = template2 | llm

    response = langchain2.invoke({"question": question, "search_results": search_results})

    output = response.content
print(output)
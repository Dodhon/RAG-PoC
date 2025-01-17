from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# pip install --upgrade --quiet duckduckgo-search
# pip install -U duckduckgo_search==5.3.1b1
# the above solved rate limit exception


# Initialize Claude model for LLM tasks
def get_llm():
    llm = ChatAnthropic(
        model='claude-3-opus-20240229',
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0
    )
    return llm


# get load, split, vectorize pdf files
def get_pdf_files(directory = '/Users/thuptenwangpo/Documents/GitHub/neo4j-practice/Sample_PDFs'):
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

    if not file_paths:
        print("No PDF files found in directory:", directory)
        exit()

    # Load and process documents
    docs = [PyMuPDFLoader(file_path).load() for file_path in file_paths]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents and create vector store
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
    )
    retriever = vectorstore.as_retriever(k=6)
    return retriever

# Get user question and retrieve relevant docs
print("")

def get_user_question():
    question = input("Please note that this is AI generated content. What is your question: ")
    return question

def get_retrieved_docs(retriever, question):
    retrieved_docs = retriever.invoke(question)
    docs_string = "".join(doc.page_content for doc in retrieved_docs)
    return docs_string

# Create initial prompt for document search
initial_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are part of an AI agent that is good at analyzing source information and answering questions.       
Use the following source documents to answer the user's questions.      
If you don't know the answer, just say 'I don't know'.       
If the answer is not directly in the documents, just say 'I don't know'. This is important for the next part of the agent.
Only use the documents to answer the question.
Do not make an answer up. If the documents don't have the answer, be concise and say 'I don't know'. Do not say anything else after that. Do not explain anything else. Just saying 'I don't know' is enough.
Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}
Answer:
""".strip()
    ),
    ("human", "{input}"),
])

# Process initial query
def process_initial_query(llm, initial_prompt, docs_string, question):
    chain = initial_prompt | llm
    response = chain.invoke({
        "docs_string": docs_string,
        "input": question,
    })
    output = response.content
    return output
# If no answer found, search the web
def check_answer(output, question):
    if output.startswith("I don't know"):
        search_web(question)
    else:
        return output

def search_web(question):
    print("")
    print("I don't know. I will now look online. Please note that the following information is ai generated.")
    print("")
    
    try:
        # Perform web search with error handling
        with DDGS(timeout=20) as duckduckgo:
            search_result = list(duckduckgo.text(
                question,  # Use original question directly
                max_results=7,
                region='wt-wt',  # Worldwide region
                safesearch='moderate'
            ))
        
        # Format search results only if we got results
        if search_result:
            search_results = []
            for result in search_result:
                search_results.append(f"""
            Title: {result['title']}
            Content: {result['body']}
            URL: {result['href']}
                """.strip())
            search_results = "\n\n".join(search_results)

            # Create prompt for web search results
            web_prompt = ChatPromptTemplate.from_template("""
            You are an AI assistant analyzing search results. Be precise and informative.
            Only use the search results below to answer. Answer the user's question based on the search results below.
            Use three sentences maximum and keep the answer concise.
            
            Search results:
            {search_results}
            User question: 
            {question}
            Answer:
            """.strip())

            web_chain = web_prompt | get_llm()
            response = web_chain.invoke({"question": question, "search_results": search_results})
            output = response.content
        else:
            output = "I apologize, but I couldn't find any relevant information online."
            
    except Exception as e:
        print(f"Error during web search: {str(e)}")
        output = "I apologize, but I encountered an error while searching online."

    print(output)

def main():
    load_dotenv()
    os.environ['USER_AGENT'] = 'USER_AGENT'
    llm = get_llm()
    retriever = get_pdf_files()
    question = get_user_question()
    docs_string = get_retrieved_docs(retriever, question)
    output = process_initial_query(llm, initial_prompt, docs_string, question)
    output = check_answer(output, question)


if __name__ == "__main__":
    main()
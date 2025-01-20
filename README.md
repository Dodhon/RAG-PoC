Hello!

This project is a proof of concept RAG that uses OpenAI llm and embeddings, Claude llm, LangChain, LangSmith, and DuckDuckGo.

To run this on your own device, you will need  OpenAI, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, and LANGCHAIN_PROJECT keys.

To demo the RAG, go to demos -> rag_demo.py.
This will first query the pdf knowledge base, then use web search grounding. 

To see the RAG metrics, you need to get langsmith keys @ https://www.langchain.com/langsmith.
As of this project, getting a free account with access is possible for devs.


demos/rag_demo_claude.py is the rag poc that uses claude for the llm. I still used openai embeddings. I've found that Claude is more verbose than instructed to be but otherwise follows prompts better.

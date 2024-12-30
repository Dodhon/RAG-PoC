from langchain_community.document_loaders import DirectoryLoader
import os
from dotenv import load_dotenv


path = "Sample_PDFs/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
dataset.to_pandas()
from datasets import Dataset, load_dataset
from langchain_community.document_loaders import PyPDFLoader
from ragas import evaluate
import os
from dotenv import load_dotenv, find_dotenv
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
import os

# Load OpenAI API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


pdf = "/Users/thuptenwangpo/Documents/GitHub/neo4j-practice/Sample_PDFs/Sample_Document_(2).pdf"
loader = PyPDFLoader(pdf)
docs = loader.load_and_split()
# print(docs[0])

from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o",
    openai_api_key=OPENAI_API_KEY
    )
                                    )
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [
    LLMContextRecall(llm=evaluator_llm), 
    FactualCorrectness(llm=evaluator_llm), 
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]
results = evaluate(dataset=eval_dataset, metrics=metrics)
df = results.to_pandas()
df.head()
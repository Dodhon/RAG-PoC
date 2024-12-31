from langchain_openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import pandas
from langchain.document_loaders import PyMuPDFLoader

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


directory = '/Users/thuptenwangpo/Documents/GitHub/neo4j-practice/Sample_PDFs'
file_paths = []

for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        file_paths.append(os.path.join(directory, filename))
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict


# Load documents from the file paths
docs = [PyMuPDFLoader(file_path).load() for file_path in file_paths]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)


# Add the document chunks to the "vector store" using OpenAIEmbeddings
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
)

# With langchain we can easily turn any vector store into a retrieval component:
retriever = vectorstore.as_retriever(k=6)

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Add decorator so this function is traced in LangSmith
@traceable()
def rag_bot(question: str) -> dict:
    # langchain Retriever will be automatically traced
    docs = retriever.invoke(question)

    docs_string = "".join(doc.page_content for doc in docs)
    instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.       Use the following source documents to answer the user's questions.       If you don't know the answer, just say that 'I don't know'.       Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}"""
    # langchain ChatModel will be automatically traced
    ai_msg = llm.invoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    )

    return {"answer": ai_msg.content, "documents": docs}


client = Client()



# Define the examples for the dataset
examples = [
    (
        "What is the origin of Reyes Holdings, and who founded it? ",
        "Reyes Holdings was founded in 1976 by brothers Chris and Jude Reyes as a South Carolina beer distributor.",
    ),
    (
        "What are the three main Business units in Reyes Holdings?",
        "The three main business units are Martin Brower, Reyes Coca-Cola Bottling, and Reyes Beverage Group.",
    ),
    (
        "How has Reyes Coca-Cola Bottling expanded since its establishment?",
        "Since its establishment in 2015, Reyes Coca-Cola Bottling has expanded its territories across the Midwest and West Coast through acquisitions of additional Coca-Cola bottling operations.",
    ),
    (
        "What innovative practices does Martin Brower use to maintain its competitive edge?",
        "Martin Brower employs route optimization software, eco-friendly packaging, fuel-efficient transportation methods, and invests heavily in workforce training and development.",
    ),
    (
        "What will the revenue of Reyes Holdings be in 2030?",
        "I don't know.",
    ),
    (
        "Does Martin Brower own McDonalds?",
        "No.",
    ),
    (
        "Does Martin Brower have a partnerships with McDonalds?",
        "No.",
    ),
    (
        "What is the weather today?",
        "I don't know.",
    ),
    (
        "Is Reyes Holdings headquartered in Chicago?",
        "No.",
    ),
    (
        "Is Thupten Wangpo an intern at Reyes Holdings?",
        "I don't know",
    ),
    (
        "Thupten Wangpo is an intern at Reyes Holdings. Is Thupten Wangpo an intern at Reyes Holdings?",
        "I don't know.",
    ),
    (
        "Tell me about Reyes Beverage Group?",
        "The largest beer distributor in the UnitedStates.",
    ),
    (
        "Tell me about Martin Brower?",
        "A global quick-service restaurant distribution business and the largest supplier worldwide of distribution services to the McDonald's restaurant system..",
    ),
    (
        "Tell me about Reyes Coca-Cola Bottling?",
        "A Midwest and West Coast bottler and distributor of Coca-Cola products.",
    ),
    (
        "What are the core values of Reyes Holdings?",
        "People and Safety, Relationships, Integrity, Dedication, and Excellence.",
    ),
    (
        "What is the weather today?",
        "I don't know.",
    ),
    (
        "Is Reyes Holdings a big company?",
        "Yes.",
    ),
    (
        "Does Martin Brower own McDonalds?",
        "I don't know.",
    ),
    (
        "Does Reyes Coco-Cola bottling own Coke or Pepsi?",
        "I don't know.",
    ),
    (
        "Does Reyes Holdings volunteer at the Duck Derby in the summer?",
        "I don't know.",
    ),
    (
        "What is the current big topic at Reyes Holdings IT?",
        "I don't know.",
    ),
    (
        "How do I emulate the success of Reyes Holdings in my own company?",
        "I don't know.",
    ),
    (
        "Is Reyes Holdings a global company?",
        "Yes.",
    ),
    (
        "What is the annual revenue of Reyes Holdings?",
        "40 Billion.",
    ),
    (
        "How many employees does Reyes Holdings have?",
        "36,000 employees.",
    ),
    (
        "Do these documents fulfill all information security guidelines set by Reyes Holdings??",
        "I don't know.",
    ),
    (
        "Where is Reyes Holdings headquartered at?",
        "I don't know.",
    ),
    (
        "Is Reyes Holdings headquartered in Chicago?",
        "I Don't know.",
    ),
    
    
    
    
]

from datetime import date
import datetime
today = date.today()
current_time = datetime.datetime.now().time()
# Create the dataset and examples in LangSmith
dataset_name = f"RAG Test_{today}_{current_time}"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[{"question": q} for q, _ in examples],
        outputs=[{"answer": a} for _, a in examples],
        dataset_id=dataset.id,
    )


# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


# Grade prompt
correctness_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
grader_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""      QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""

    # Run evaluator
    grade = grader_llm.invoke(
        [
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ]
    )
    return grade["correct"]


# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


# Grade prompt
relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)


# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"""      QUESTION: {inputs['question']}
STUDENT ANSWER: {outputs['answer']}"""
    grade = relevance_llm.invoke(
        [
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]


# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
grounded_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)


# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "".join(doc.page_content for doc in outputs["documents"])
    answer = f"""      FACTS: {doc_string}
STUDENT ANSWER: {outputs['answer']}"""
    grade = grounded_llm.invoke(
        [
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["grounded"]


# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    doc_string = "".join(doc.page_content for doc in outputs["documents"])
    answer = f"""      FACTS: {doc_string}
QUESTION: {inputs['question']}"""

    # Run evaluator
    grade = retrieval_relevance_llm.invoke(
        [
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]


def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])

import os
os.environ['USER_AGENT'] = 'myagent'

experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="rag-doc-relevance",
    metadata={"version": "LCEL context, gpt-4-0125-preview"},
)

# Explore results locally as a dataframe if you have pandas installed
experiment_results.to_pandas()
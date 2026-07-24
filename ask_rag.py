import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load FAISS database
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


question = input("\nAsk question: ")


# Retrieve relevant chunks
docs = db.similarity_search(question, k=3)


context = "\n\n".join(
    [doc.page_content for doc in docs]
)


prompt = f"""
You are a healthcare research assistant.

Answer only using the given research paper context.

Context:
{context}

Question:
{question}

Answer:
"""


response = llm.invoke(prompt)


print("\nAnswer:")
print(response.content)
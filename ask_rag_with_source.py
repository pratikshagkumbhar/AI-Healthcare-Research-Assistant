import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


question = input("\nAsk question: ")


docs = db.similarity_search(question, k=3)


context = "\n\n".join(
    [doc.page_content for doc in docs]
)


prompt = f"""
You are an AI Healthcare Research Assistant.

Use only the given context.

Context:
{context}

Question:
{question}

Answer clearly with technical details.
"""


response = llm.invoke(prompt)


print("\nAnswer:")
print(response.content)


print("\nSources:")
for i, doc in enumerate(docs):
    print("\nSource", i+1)
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:200])
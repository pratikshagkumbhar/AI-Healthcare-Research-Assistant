from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = "papers/ecg_paper.pdf"


loader = PyPDFLoader(pdf_path)

documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)


print("Chunks:", len(chunks))


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


vectorstore.save_local("faiss_index")


print("Vector database created!")
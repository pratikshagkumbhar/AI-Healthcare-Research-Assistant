import streamlit as st
import tempfile

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


st.set_page_config(
    page_title="AI Healthcare Research Assistant",
    page_icon="🩺",
    layout="wide"
)


st.title("🩺 AI Healthcare Research Assistant")

st.write(
    "Upload multiple healthcare research papers and ask questions using RAG."
)


# ---------------- MODELS ----------------


@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )

embeddings = load_embeddings()
llm = load_llm()

def get_text(response):
    if isinstance(response.content, str):
        return response.content

    elif isinstance(response.content, list):
        text = ""

        for item in response.content:
            if isinstance(item, dict):
                text += item.get("text", "")
            elif hasattr(item, "text"):
                text += item.text
            else:
                text += str(item)

        return text

    return str(response.content)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- PDF UPLOAD ----------------


uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF)",
    type="pdf",
    accept_multiple_files=True
)


if uploaded_files:

    if st.button("Process Papers"):

        all_documents = []


        with st.spinner("Reading research papers..."):

            for uploaded_file in uploaded_files:


                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as f:

                    f.write(
                        uploaded_file.getvalue()
                    )

                    pdf_path = f.name


                loader = PyPDFLoader(pdf_path)

                documents = loader.load()


                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name


                all_documents.extend(documents)



        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )


        chunks = splitter.split_documents(
            all_documents
        )


        db = FAISS.from_documents(
            chunks,
            embeddings
        )


        st.session_state["db"] = db


        st.success(
            f"{len(uploaded_files)} papers processed successfully!"
        )



# ---------------- FEATURES ----------------


if "db" in st.session_state:


    db = st.session_state["db"]


    # -------- Q&A --------

    st.header("Ask Research Questions")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):

    if question:

        docs = db.similarity_search(question, k=6)

        context = "\n\n".join(
            [d.page_content for d in docs]
        )

        prompt = f"""
You are an AI Healthcare Research Assistant.

Answer only from the given research paper context.

Provide:
- Direct answer
- Technical explanation
- Preprocessing/model details

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        st.subheader("Answer")

        answer = get_text(response)

        st.markdown(answer)

        st.session_state.messages.append(
            {
                "q": question,
                "a": answer
            }
        )

        st.subheader("Sources")

        for i, doc in enumerate(docs):
            st.write(
                f"""
Source {i+1}

Paper: {doc.metadata.get('source')}

Page: {doc.metadata.get('page') + 1}
"""
            )

        # -------- SUMMARY --------


st.divider()

if st.button("Summarize Research Papers"):

    docs = db.similarity_search(
        "objective dataset preprocessing model results conclusion",
        k=6
    )

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Summarize these healthcare research papers.

Include:
1. Objective
2. Dataset
3. Preprocessing techniques
4. Model architecture
5. Results
6. Limitations

Context:
{context}
"""

    result = llm.invoke(prompt)

    st.subheader("Research Paper Summary")

    st.markdown(get_text(result))


    # -------- COMPARISON --------

st.divider()

compare = st.text_input("Compare papers:")

if st.button("Compare Papers"):
    docs = db.similarity_search(compare, k=8)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Compare the research papers based on:

- Dataset
- Preprocessing
- Model architecture
- Performance
- Advantages
- Limitations

Context:
{context}

Question:
{compare}
"""

    result = llm.invoke(prompt)
    st.subheader("Paper Comparison")
    st.markdown(get_text(result))

    # -------- TERM EXPLANATION --------

st.divider()

term = st.text_input("Explain medical/AI term:")

if st.button("Explain Term"):
    prompt = f"""
Explain this term:

{term}

Give:
1. Simple explanation
2. Technical definition
3. Healthcare application
"""

    result = llm.invoke(prompt)
    st.subheader("Explanation")
    st.markdown(get_text(result))

    st.subheader("Explanation")

    st.markdown(get_text(result))


            # -------- CHAT HISTORY --------

    st.divider()

    st.subheader("Chat History")

    for chat in st.session_state.messages:

        st.write("Question:", chat["q"])
        st.write("Answer:", chat["a"])
        st.write("---")
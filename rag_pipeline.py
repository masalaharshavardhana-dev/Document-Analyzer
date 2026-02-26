from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import tempfile
import os 
from dotenv import load_dotenv

load_dotenv()

def run_rag(file, question):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        file_path = tmp.name

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    api_key = os.getenv("GROQ_API_KEY")
    model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=api_key,
    )
    vector_store = FAISS.from_documents(
        docs,
        embedding_model
    )
    retrieved_docs = vector_store.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    system_prompt = """
    You are an intelligent document analysis assistant.

    Answer the user's question strictly using the provided context.
    If the answer is not found in the context, say:
    "The document does not contain enough information to answer this question."

    Do not make up information.
    Keep the answer clear and concise.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]

    response = model.invoke(messages)

    return response.content


    
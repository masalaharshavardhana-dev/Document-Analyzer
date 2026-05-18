import streamlit as st
from router import route_document

st.title("📄 Document Analyzer")

uploaded_file = st.file_uploader("Upload your document", 
                                  type=["pdf", "txt", "docx", "csv", "xlsx"])

if uploaded_file:
    user_question = st.text_input("Ask a question about the document")

    if user_question:
        answer = route_document(uploaded_file, user_question)
        st.write("### 🤖 Answer:")
        st.write(answer)
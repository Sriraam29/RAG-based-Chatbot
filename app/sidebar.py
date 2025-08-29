import streamlit as st
import os

def display_sidebar():
    # Sidebar: Model Selection (Ollama models available locally)
    model_options = ["mistral", "llama2", "codellama", "gemma"]  # Add whichever you pulled with ollama
    st.sidebar.selectbox("Select Ollama Model", options=model_options, key="model")

    # Sidebar: Upload Document
    st.sidebar.header("Upload Document for RAG")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt", "html","csv","xlsx"])

    if uploaded_file is not None:
        save_path = os.path.join("uploaded_docs", uploaded_file.name)
        os.makedirs("uploaded_docs", exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"Saved {uploaded_file.name} to local folder.")
        # Optional: Trigger embedding update
        st.session_state.new_doc_uploaded = True  

    # Sidebar: Show Uploaded Documents
    st.sidebar.header("Uploaded Documents")
    if os.path.exists("uploaded_docs"):
        files = os.listdir("uploaded_docs")
        if files:
            for f in files:
                st.sidebar.text(f)
        else:
            st.sidebar.info("No documents uploaded yet.")

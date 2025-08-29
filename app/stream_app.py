# Updated streamlit_app.py (assuming this is your frontend)
# Use get_api_response from api_utils
# Display context for debug
# Add file upload/list/delete in sidebar

import streamlit as st
from api_utils import get_api_response, upload_document, list_documents, delete_document
import uuid

st.title("VPRC RAG Chatbot")

# Model selection
model_options = ["llama2", "mistral"]  # Adjust based on your Ollama models
selected_model = st.sidebar.selectbox("Select Model", model_options)

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    response = get_api_response(prompt, st.session_state.session_id, selected_model)

    if response:
        answer = response["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        st.error("Failed to get response from backend.")

# Sidebar for document management
st.sidebar.header("Document Management")

# Upload
uploaded_file = st.sidebar.file_uploader("Upload Document", type=["pdf", "docx", "txt", "html"])
if uploaded_file:
    upload_result = upload_document(uploaded_file)
    if upload_result:
        st.sidebar.success(f"Uploaded and indexed: {uploaded_file.name}")

# List documents
docs = list_documents()
if docs:
    st.sidebar.subheader("Uploaded Documents")
    for doc in docs:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(doc["filename"])
        if col2.button("Delete", key=f"del_{doc['id']}"):
            delete_result = delete_document(doc["id"])
            if delete_result:
                st.sidebar.success(f"Deleted {doc['filename']}")
                st.rerun()  # Refresh list
else:
    st.sidebar.info("No documents uploaded.")
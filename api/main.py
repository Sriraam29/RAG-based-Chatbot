import os
import uuid
import shutil
import logging
from typing import Any, List, Union, Dict
import time


from fastapi import FastAPI, File, UploadFile, HTTPException,Form
from fastapi.middleware.cors import CORSMiddleware

from pydantic_models import (
    QueryInput,
    QueryResponse,
    DocumentInfo,
    DeleteFileRequest,
)
from langchain_utils import get_rag_chain
from db_utils import (
    insert_application_logs,
    get_chat_history,
    get_all_documents,
    insert_document_record,
    delete_document_record,
)
from chroma_util import index_document_to_chroma, reset_chroma,similarity_search

# -------------------------------------
# App & logging
# -------------------------------------
logging.basicConfig(filename="app.log", level=logging.INFO)
app = FastAPI(title="RAG API (Ollama + Chroma)")


# -------------------------------------
# Helpers
# -------------------------------------


def _normalize_history(raw_history: Any) -> List[Dict[str, str]]:
    """
    Normalize chat history to a standard format.
    """
    msgs = []
    if not raw_history:
        return msgs

    # Handle different message formats
    for msg in raw_history:
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            # Handle LangChain message objects
            role = "user" if msg.type in ["human", "user"] else "assistant"
            msgs.append({"role": role, "content": msg.content})
        elif isinstance(msg, dict):
            # Handle dictionary format
            if "role" in msg and "content" in msg:
                msgs.append(msg)
            elif "question" in msg and "answer" in msg:
                msgs.append({"role": "user", "content": msg["question"]})
                msgs.append({"role": "assistant", "content": msg["answer"]})
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            # Handle (role, content) tuples
            role, content = msg
            msgs.append({"role": role, "content": content})
    
    return msgs

def _model_name(model_field: Union[str, Any]) -> str:
    """
    Your pydantic now uses `model: str` with default 'mistral'.
    If you ever switch back to Enum, this still handles `.value`.
    """
    try:
        return model_field.value  # Enum-like
    except AttributeError:
        return str(model_field)


# -------------------------------------
# Chat endpoint
# -------------------------------------
@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    model_name = _model_name(query_input.model)

    logging.info(
        f"[CHAT] session={session_id} model={model_name} question={query_input.question!r}"
    )

    # Load & normalize conversation history
    raw_history = get_chat_history(session_id)
    chat_history = _normalize_history(raw_history)

    # Build RAG chain (history-aware in your langchain_utils)
    rag_chain = get_rag_chain(model_name)

    # Run query
    try:
        response = rag_chain.invoke({"input": query_input.question, "chat_history": chat_history})
        # Try common keys from different chain return shapes
        answer = (
            (response.get("answer") if isinstance(response, dict) else None)
            or (response.get("result") if isinstance(response, dict) else None)
            or (response.get("output") if isinstance(response, dict) else None)
            or (response if isinstance(response, str) else None)
        )
        if not isinstance(answer, str):
            answer = str(response)
    except Exception as e:
        logging.exception(f"[CHAT] error session={session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating response.")

    # Log to DB
    try:
        insert_application_logs(session_id, query_input.question, answer, model_name)
    except Exception as e:
        logging.exception(f"[CHAT] log insert failed session={session_id}: {e}")

    logging.info(f"[CHAT] session={session_id} -> {answer!r}")

    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


# -------------------------------------
# Upload & index docs
# -------------------------------------
@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = [".pdf", ".docx", ".html", ".txt", ".md",".csv",".xlsx"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}",
        )

    temp_file_path = f"temp_{file.filename}"

    try:
        # Save file to disk temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Record in DB
        file_id = insert_document_record(file.filename)

        # Index to Chroma (handle Ollama embedding errors gracefully)
        try:
            result = index_document_to_chroma(temp_file_path, file_id)
        except Exception as e:
            # Roll back DB record on hard failure
            delete_document_record(file_id)
            logging.exception(f"[UPLOAD] hard index failure for {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail="Indexing failed (embedding server error). Ensure Ollama is running and embedding model is pulled.",
            )

        if isinstance(result, dict) and result.get("ok"):
            chunks = result.get("chunks", 0)
            return {
                "message": f"File {file.filename} indexed ({chunks} chunks).",
                "file_id": file_id,
                "chunks": chunks,
            }

        # If index call returned a falsy status
        delete_document_record(file_id)
        logging.error(f"[UPLOAD] indexing returned not ok for {file.filename}: {result}")
        raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# -------------------------------------
# List docs
# -------------------------------------
@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


# -------------------------------------
# Delete doc
# -------------------------------------
@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id}."}
        else:
            return {
                "error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."
            }
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}


# -------------------------------------
# Local run convenience
# -------------------------------------
if __name__ == "__main__":
    import uvicorn
    # host 0.0.0.0 lets docker/other machines connect; change to 127.0.0.1 if you prefer local only
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
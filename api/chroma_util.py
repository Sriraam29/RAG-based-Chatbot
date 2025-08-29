# chroma_util.py (auto-reset, safe, no eager globals)

from __future__ import annotations

import os
import shutil
import stat
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Union

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    UnstructuredExcelLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# Try to import Chroma with fallback for different versions
try:
    from langchain_chroma import Chroma
    CHROMA_NEW = True
except ImportError:
    try:
        from langchain.vectorstores import Chroma
        CHROMA_NEW = False
    except ImportError:
        raise ImportError("Could not import Chroma from either langchain_chroma or langchain.vectorstores")

# -----------------------------
# Config (env overridable)
# -----------------------------
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "default")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Lock to avoid races while wiping/recreating the DB
_VS_RESET_LOCK = threading.Lock()

# -----------------------------
# Splitter / Embeddings
# -----------------------------
@lru_cache(maxsize=1)
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )

@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """
    Uses Ollama for embeddings (e.g., model='nomic-embed-text').
    Ensure your Ollama server is running and the embed model is pulled:
      ollama pull nomic-embed-text
    """
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Lazy singleton vectorstore. Do NOT call at module import.
    Recreated after reset_chroma() clears this cache.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    if CHROMA_NEW:
        return Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
        )
    else:
        vs = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
        )
        os.makedirs(CHROMA_DIR, exist_ok=True)
        return vs

# -----------------------------
# Helpers
# -----------------------------
def _on_rm_error(func, path, exc_info):
    # Make read-only files writable then retry (Windows-friendly)
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def reset_chroma() -> None:
    """
    Clears the entire Chroma persistence directory and rebuilds a fresh vectorstore.
    Also clears the lru_cache so a brand-new instance is created on next use.
    """
    with _VS_RESET_LOCK:
        # 1) Drop the folder
        shutil.rmtree(CHROMA_DIR, ignore_errors=False, onerror=_on_rm_error)
        # 2) Clear cached instances so new ones are created against the fresh dir
        try:
            get_vectorstore.cache_clear()
        except Exception:
            pass
        # Ensure dir exists for the next instantiation
        os.makedirs(CHROMA_DIR, exist_ok=True)

# -----------------------------
# Loading / splitting
# -----------------------------
def _pick_loader(file_path: str):
    fp = file_path.lower()
    if fp.endswith(".pdf"):
        return PyPDFLoader(file_path)
    if fp.endswith(".docx"):
        return Docx2txtLoader(file_path)
    if fp.endswith(".html") or fp.endswith(".htm"):
        return UnstructuredHTMLLoader(file_path)
    if fp.endswith(".txt") or fp.endswith(".md"):
        return TextLoader(file_path, autodetect_encoding=True)
    if fp.endswith(".xlsx"):
        return UnstructuredExcelLoader(file_path, mode="elements")
    if fp.endswith(".csv"):
        return CSVLoader(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")

def load_and_split_document(file_path: str) -> List[Document]:
    docs = _pick_loader(file_path).load()
    if not docs:
        return []
    return get_text_splitter().split_documents(docs)

# -----------------------------
# Indexing / deletion / search
# -----------------------------
def index_document_to_chroma(
    file_path: str,
    file_id: Union[int, str],
    extra_metadata: Optional[Dict] = None,
) -> Dict[str, Union[bool, int, str]]:
    """
    Index a document into Chroma with metadata {file_id, ...extra_metadata}.
    Returns: {"ok": bool, "chunks": int, "message": str}
    """
    try:
        splits = load_and_split_document(file_path)
        if not splits:
            return {"ok": False, "chunks": 0, "message": "No content extracted from document."}

        # Attach metadata
        for split in splits:
            md = dict(split.metadata or {})
            md["file_id"] = str(file_id)
            if extra_metadata:
                md.update(extra_metadata)
            split.metadata = md

        vs = get_vectorstore()
        vs.add_documents(splits)
        if not CHROMA_NEW:
            vs.persist()

        return {"ok": True, "chunks": len(splits), "message": "Indexed successfully."}
    except Exception as e:
        print(f"[chroma_util] Error indexing document ({file_path}): {e}")
        return {"ok": False, "chunks": 0, "message": str(e)}






def similarity_search(
    query: str,
    k: int = 4,
) -> List[Document]:
    return get_vectorstore().similarity_search(query, k=k)

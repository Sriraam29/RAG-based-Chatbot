# Updated langchain_utils.py to match notebook's conversational RAG
# Use Chroma instead of FAISS for consistency with chroma_util.py
# Build full conversational chain with history-aware retriever

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chroma_util import get_vectorstore  # Import global Chroma

def get_rag_chain(model_name="llama2"):
    """
    Build conversational RAG chain using Ollama LLM and Chroma retriever.
    Matches the notebook's structure.
    """
    # LLM from Ollama
    llm = ChatOllama(model=model_name, temperature=0.7)

    # Retriever from global Chroma (populated via uploads)
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 2})

    # Contextualize query prompt (for history)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # History-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA prompt (uses context)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say 'I don't have information about that in the provided documents.'"),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Stuff chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full conversational RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
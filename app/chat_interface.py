import streamlit as st
import ollama  # pip install ollama

def get_ollama_response(prompt, model="llama2"):
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def display_chat_interface():
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Query:"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from Ollama
        with st.spinner("Generating response with Ollama..."):
            response = get_ollama_response(prompt, st.session_state.model)

            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

                    with st.expander("Details"):
                        st.subheader("Generated Answer")
                        st.code(response)
                        st.subheader("Model Used")
                        st.code(st.session_state.model)
            else:
                st.error("❌ Failed to generate response. Please check Ollama server.")

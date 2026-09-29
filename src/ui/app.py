import os
import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/query")

st.title("🤖 Enterprise IT Support Assistant")
st.caption("Powered by Local Hybrid Search (MySQL + ChromaDB) & Ollama (Llama 3.2)")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat Messages from History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Field
if prompt := st.chat_input("Ask about IT issues, VPN, Router, or Printer troubleshooting..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI Endpoint
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 *Searching knowledge base and generating answer...*")
        
        try:
            response = requests.post(
                API_URL,
                json={"query": prompt},
                timeout=120
            )
            
            if response.status_code == 200:
                answer = response.json().get("response", "No response content received.")
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"❌ Error {response.status_code}: Could not fetch response from API."
                message_placeholder.error(error_msg)
        except Exception as e:
            message_placeholder.error(f"❌ Failed to connect to API server: {str(e)}")
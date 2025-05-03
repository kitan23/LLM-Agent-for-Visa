import streamlit as st
import requests
import os
import json

# API URL - either from environment variable or default to localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="OPT-RAG Assistant",
    page_icon="🎓",
    layout="wide",
)

# Page title and description
st.title("🎓 OPT-RAG: International Student Visa Assistant")
st.markdown("""
This assistant helps international students navigate visa-related issues, 
OPT applications, and other immigration concerns using 
Retrieval-Augmented Generation (RAG) technology.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    OPT-RAG provides accurate information by retrieving 
    relevant content from official documentation and policies.
    """)
    
    st.header("Features")
    st.markdown("""
    - OPT/CPT Application Help
    - Visa Status Questions
    - Work Authorization Info
    - Document Requirements
    """)
    
    # System status
    st.header("System Status")
    
    try:
        # Check health endpoint
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            st.success("System Online ✅")
        else:
            st.error("System Offline ❌")
    except Exception as e:
        st.error(f"Cannot connect to API: {e}")

# Chat interface
st.header("Ask Your Question")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("What visa question can I help with today?")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Searching visa regulations...")
        
        try:
            # Send request to API - use streaming endpoint
            with st.spinner("Generating response..."):
                response = requests.post(
                    f"{API_URL}/api/query",
                    json={"question": prompt}
                )
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer provided")
                    message_placeholder.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    message_placeholder.markdown(f"❌ {error_msg}")
        except Exception as e:
            message_placeholder.markdown(f"❌ Error: {str(e)}")

# Document uploader section
st.header("Document Management")
st.markdown("Upload official documents to enhance the knowledge base.")

uploaded_file = st.file_uploader("Upload Immigration Document", type=["pdf", "txt", "docx"])

if uploaded_file and st.button("Add Document"):
    # Save uploaded file temporarily
    file_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Call API to add document
    try:
        with st.spinner("Processing document..."):
            files = {"file": (uploaded_file.name, open(file_path, "rb"))}
            response = requests.post(f"{API_URL}/api/documents", files=files)
            
            if response.status_code == 200:
                st.success(f"Document added: {uploaded_file.name}")
            else:
                st.error(f"Error adding document: {response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

# Document list
st.subheader("Current Documents")
try:
    response = requests.get(f"{API_URL}/api/documents")
    if response.status_code == 200:
        documents = response.json().get("documents", [])
        if documents:
            for doc in documents:
                st.markdown(f"- **{doc['source']}** ({doc['document_type']}) - {doc['chunk_count']} chunks")
        else:
            st.info("No documents in the system yet. Upload your first document above.")
    else:
        st.warning("Could not retrieve document list.")
except Exception as e:
    st.warning(f"Could not connect to API: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Disclaimer**: This assistant provides information based on available documents. It is not a substitute for legal advice.") 
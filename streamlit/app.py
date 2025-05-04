import streamlit as st
import requests
import os
import json
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger("streamlit-app")

# API URL - either from environment variable or default to localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000")
# logger.info(f"Using API URL: {API_URL}")

# Display the environment variables for debugging
logger.info(f"Environment variables: {dict(os.environ)}")

st.set_page_config(
    page_title="OPT-RAG Assistant",
    page_icon="🎓",
    layout="wide",
)

# Debug information at the top (will remove later)
st.info(f"API URL: {API_URL}")

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
        logger.info(f"Checking health at {API_URL}/health")
        response = requests.get(f"{API_URL}/health", timeout=5)
        logger.info(f"Health check response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            st.success("System Online ✅")
        else:
            st.error(f"System Offline ❌ (Status: {response.status_code})")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        st.error(f"Cannot connect to API: {str(e)}")

# Initialize uploaded documents list in session state if not present
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

# Document uploader section
st.header("Document Management")
st.markdown("Upload official documents to enhance the knowledge base.")

uploaded_file = st.file_uploader("Upload Immigration Document", type=["pdf", "txt", "docx"])

if uploaded_file and st.button("Add Document"):
    logger.info(f"Uploading document: {uploaded_file.name}")
    # Save uploaded file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Call API to add document
    try:
        with st.spinner("Processing document..."):
            # Create a multipart form with the file
            files = {"file": (uploaded_file.name, open(file_path, "rb"), "application/octet-stream")}
            form_data = {"document_type": "immigration"}
            
            # Send the request
            response = requests.post(
                f"{API_URL}/documents", 
                files=files,
                data=form_data,
                timeout=60
            )
            
            logger.info(f"Document upload response: {response.status_code}")
            
            if response.status_code == 200:
                # Add document to session state
                doc_info = {
                    "source": uploaded_file.name,
                    "document_type": "immigration",
                    "timestamp": st.session_state.get("_timestamp", "")
                }
                st.session_state.uploaded_documents.append(doc_info)
                st.success(f"Document added: {uploaded_file.name}")
            else:
                st.error(f"Error adding document: {response.text}")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        st.error(f"Error: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

# Document list
st.subheader("Your Uploaded Documents")
if st.session_state.uploaded_documents:
    for doc in st.session_state.uploaded_documents:
        st.markdown(f"- **{doc['source']}** ({doc['document_type']})")
else:
    st.info("No documents uploaded yet. Upload your first document above.")

# Add a divider between document management and chat
st.markdown("---")

# Chat interface
st.header("Chat with the Assistant")

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
    logger.info(f"Received prompt: {prompt}")
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
            logger.info(f"Sending query to {API_URL}/api/query")
            with st.spinner("Generating response..."):
                response = requests.post(
                    f"{API_URL}/api/query",
                    json={"question": prompt},
                    timeout=60  # Increased timeout
                )
                
                logger.info(f"API response status: {response.status_code}")
                if response.status_code == 200:
                    try:
                        answer = response.json().get("answer", "No answer provided")
                        logger.info(f"Got answer: {answer[:50]}...")
                        message_placeholder.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        logger.error(f"Error parsing response: {e}")
                        message_placeholder.markdown(f"❌ Error parsing response: {str(e)}")
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    message_placeholder.markdown(f"❌ {error_msg}")
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            message_placeholder.markdown(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Disclaimer**: This assistant provides information based on available documents. It is not a substitute for legal advice.") 
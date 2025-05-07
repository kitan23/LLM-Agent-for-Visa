import streamlit as st
import requests
import os
import json
import logging
import sys
import threading
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger("streamlit-app")

# API URL - either from environment variable or default to localhost
RAW_API_URL = os.environ.get("API_URL", "http://localhost:8000")
logger.info(f"Raw API URL from environment: {RAW_API_URL}")

# API helper functions
def get_api_url(endpoint):
    """
    Constructs properly formatted API URLs, handling the /api prefix correctly.
    
    Args:
        endpoint: The API endpoint (without leading slash)
        
    Returns:
        Properly formatted full URL
    """
    # Strip any trailing slashes from the base URL
    base_url = RAW_API_URL.rstrip('/')
    
    # If endpoint already starts with /api, don't add it again
    if endpoint.startswith('/'):
        endpoint = endpoint[1:]
    
    # If base URL already contains /api and endpoint is meant to be under /api
    if '/api' in base_url:
        # For endpoints that should be under /api
        full_url = f"{base_url}/{endpoint}"
    else:
        # Need to add /api prefix
        full_url = f"{base_url}/api/{endpoint}"
    
    logger.info(f"Constructed API URL: {full_url}")
    return full_url

# Display the environment variables for debugging
logger.info(f"Environment variables: {dict(os.environ)}")

st.set_page_config(
    page_title="OPT-RAG Assistant",
    page_icon="🎓",
    layout="wide",
)

# Debug information at the top (will remove later)
st.info(f"API URL: {RAW_API_URL}")

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
        # Check health endpoint using the helper
        health_url = get_api_url('health')
        logger.info(f"Checking health at {health_url}")
        
        response = requests.get(health_url, timeout=5)
        logger.info(f"Health check response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            st.success("System Online ✅")
        else:
            st.error(f"System Offline ❌ (Status: {response.status_code})")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        st.error(f"Cannot connect to API: {str(e)}")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "should_stop_generation" not in st.session_state:
    st.session_state.should_stop_generation = False

if "current_request_id" not in st.session_state:
    st.session_state.current_request_id = None

# Function to handle stopping generation
def stop_generation():
    # First, set the local flag to stop displaying more tokens
    st.session_state.should_stop_generation = True
    logger.info("User requested to stop generation")
    
    # Then, if we have a request ID, call the API to cancel the generation
    if st.session_state.current_request_id:
        try:
            logger.info(f"Attempting to cancel generation for request ID: {st.session_state.current_request_id}")
            
            # Get cancel URL using the helper
            cancel_url = get_api_url('cancel')
            logger.info(f"Sending cancellation request to {cancel_url}")
            
            cancel_response = requests.post(
                cancel_url,
                json={"request_id": st.session_state.current_request_id},
                timeout=5,
                headers={"Connection": "close"}  # Ensure connection closes after request
            )
            
            # Log detailed response for debugging
            status_code = cancel_response.status_code
            logger.info(f"Cancel request response status: {status_code}")
            
            if status_code == 200:
                logger.info(f"Successfully sent cancellation request for {st.session_state.current_request_id}")
            else:
                logger.warning(f"Failed to cancel generation: {status_code} - {cancel_response.text}")
        except Exception as e:
            logger.error(f"Error sending cancellation request: {str(e)}", exc_info=True)

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
            
            # Get URL using the helper
            documents_url = get_api_url('documents')
            logger.info(f"Sending document to {documents_url}")
            
            # Send the request
            response = requests.post(
                documents_url, 
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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("What visa question can I help with today?", disabled=st.session_state.is_generating)

if prompt:
    logger.info(f"Received prompt: {prompt}")
    # Reset stop flag when starting new generation
    st.session_state.should_stop_generation = False
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        message_placeholder.markdown("🤔 Searching visa regulations...")
        
        # Create a column layout for the stop button
        cols = st.columns([9, 1])
        
        # Show stop button in the small column during generation
        with cols[1]:
            stop_button = st.button("Stop", key="stop_button", on_click=stop_generation, disabled=False)
        
        # Mark that we're generating - disables the input prompt 
        st.session_state.is_generating = True
        
        try:
            # Use streaming endpoint with helper function
            stream_url = get_api_url('query/stream')
            logger.info(f"Sending streaming query to {stream_url}")
            
            # Set a long timeout for streaming
            timeout = 300  # 5 minutes
            
            # Initialize SSE client session
            with requests.post(
                stream_url,
                json={"question": prompt},
                stream=True,
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                },
                timeout=timeout
            ) as response:
                logger.info(f"Streaming API response status: {response.status_code}")
                
                if response.status_code == 200:
                    # Check for request ID in headers
                    request_id = response.headers.get("X-Request-ID")
                    if request_id:
                        logger.info(f"Received request ID: {request_id}")
                        st.session_state.current_request_id = request_id
                    
                    # Process the SSE stream
                    response_started = False
                    
                    # Track partial SSE data
                    buffer = ""
                    
                    # Process the response as a stream of bytes
                    for chunk in response.iter_content(chunk_size=1024):
                        # Check if user clicked stop button
                        if st.session_state.should_stop_generation:
                            logger.info("Stopping generation as requested by user")
                            break
                        
                        if not chunk:
                            continue
                            
                        # Decode chunk and add to buffer
                        buffer += chunk.decode('utf-8')
                        
                        # Process complete SSE messages
                        while '\n\n' in buffer:
                            message, buffer = buffer.split('\n\n', 1)
                            
                            # Find data lines
                            for line in message.split('\n'):
                                if line.startswith('data:'):
                                    data = line[5:].strip()
                                    
                                    # Check if we've reached the end
                                    if data == '[DONE]':
                                        logger.info("Streaming complete")
                                        break
                                        
                                    try:
                                        # Parse the JSON-formatted data
                                        parsed_data = json.loads(data)
                                        
                                        # Check if this is the initial message with request ID
                                        if isinstance(parsed_data, dict) and "request_id" in parsed_data:
                                            st.session_state.current_request_id = parsed_data["request_id"]
                                            logger.info(f"Stored request ID from streaming data: {parsed_data['request_id']}")
                                            continue
                                        
                                        # Check if this is a status message
                                        if isinstance(parsed_data, dict) and "status" in parsed_data:
                                            if parsed_data["status"] == "cancelled":
                                                logger.info("Received cancellation confirmation from server")
                                                continue
                                        
                                        # Treat as a normal token
                                        token = parsed_data
                                        response_started = True
                                        
                                        # Log token received
                                        if token_count := token_count + 1 if 'token_count' in locals() else 1:
                                            if token_count % 20 == 0:
                                                logger.info(f"Received {token_count} tokens so far")
                                        
                                        # Skip common assistant prefixes
                                        if token in ["A:", "A: ", "Assistant:", "Assistant: ", "AI:", "AI: ", "Human:", "Human: "]:
                                            logger.info(f"Skipping prefix token: {token}")
                                            continue
                                        
                                        # Add the token to our response
                                        full_response += token
                                        
                                        # Update the UI with accumulated tokens
                                        # Add a blinking cursor to show it's streaming
                                        message_placeholder.markdown(full_response + "▌")
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Error parsing token: {e}, data: {data!r}")
                                        # Try to recover - if it looks like a valid string anyway, use it
                                        if data.startswith('"') and data.endswith('"'):
                                            try:
                                                token = data.strip('"')
                                                full_response += token
                                                message_placeholder.markdown(full_response + "▌")
                                                response_started = True
                                                logger.info(f"Recovered token from malformed JSON: {token}")
                                            except Exception:
                                                pass
                    
                    # Check if we actually got any response
                    if not response_started and not st.session_state.should_stop_generation:
                        logger.error("No tokens received from the streaming endpoint")
                        # Fall back to non-streaming API
                        logger.info("Falling back to non-streaming endpoint")
                        
                        # Use helper for fallback URL
                        fallback_url = get_api_url('query')
                        logger.info(f"Using fallback endpoint: {fallback_url}")
                        
                        fallback_response = requests.post(
                            fallback_url,
                            json={"question": prompt},
                            timeout=timeout
                        )
                        
                        if fallback_response.status_code == 200:
                            answer = fallback_response.json().get("answer", "No answer provided")
                            # Clean prefixes
                            for prefix in ["A:", "Assistant:", "AI:"]:
                                if answer.startswith(prefix):
                                    answer = answer[len(prefix):].lstrip()
                                    break
                                
                            message_placeholder.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            logger.info("Successfully used fallback non-streaming endpoint")
                        else:
                            message_placeholder.markdown("❌ No tokens received and fallback also failed.")
                    elif st.session_state.should_stop_generation:
                        # If generation was stopped, add message
                        message_placeholder.markdown(full_response + " *(Generation stopped)*")
                        st.session_state.messages.append({"role": "assistant", "content": full_response + " *(Generation stopped)*"})
                        logger.info("Response marked as stopped by user")
                    else:
                        # Final cleanup of the response - remove any remaining prefixes
                        for prefix in ["A:", "Assistant:", "AI:"]:
                            if full_response.startswith(prefix):
                                full_response = full_response[len(prefix):].lstrip()
                                break
                            
                        # Final update without the cursor
                        message_placeholder.markdown(full_response)
                        
                        # Add final response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        logger.info(f"Completed streaming response of length {len(full_response)}")
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    message_placeholder.markdown(f"❌ {error_msg}")
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {timeout} seconds")
            message_placeholder.markdown("❌ Request timed out. The server took too long to respond.")
        except Exception as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            message_placeholder.markdown(f"❌ Error: {str(e)}")
        finally:
            # Re-enable the chat input
            st.session_state.is_generating = False
            # Reset the stop button flag 
            st.session_state.should_stop_generation = False
            # Clear the request ID
            st.session_state.current_request_id = None

# Footer
# st.markdown("---")
# st.markdown("**Disclaimer**: This assistant provides information based on available documents. It is not a substitute for legal advice.") 
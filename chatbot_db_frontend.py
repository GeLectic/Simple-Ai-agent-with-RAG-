import streamlit as st
import uuid
import tempfile
import os

# Import the new ingest_document function along with your chatbot
from langgraph_tool_backend import chatbot, retrieve_all_threads, ingest_document
from langchain_core.messages import AIMessage, HumanMessage

#****************************** Utility Functions *******************************

def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state_values = chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values
    return state_values.get('messages', [])

#**************** Session State Initialization ****************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

# NEW: Track uploaded files to prevent redundant vectorization on every Streamlit rerun
if 'uploaded_filename' not in st.session_state:
    st.session_state['uploaded_filename'] = None

add_thread(st.session_state['thread_id'])

#********************************* Sidebar UI *********************************

st.sidebar.title("LangGraph Chatbot")

# === File Uploader Integration ===
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF to chat with", type=["pdf"])

if uploaded_file:
    if st.session_state['uploaded_filename'] != uploaded_file.name:
        with st.sidebar.status("Processing PDF...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            ingest_document(tmp_path)
            os.remove(tmp_path)
            st.session_state['uploaded_filename'] = uploaded_file.name
            status.update(label="Document vectorized & ready!", state="complete", expanded=False)

st.sidebar.divider()

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header("My Conversations")

# Add a dictionary to store titles so we don't query the DB constantly
if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

for thread_id in st.session_state['chat_threads'][::-1]:
    
    # Fetch or generate the chat title based on the first user message
    if thread_id not in st.session_state['thread_titles'] or st.session_state['thread_titles'][thread_id] == "New Chat":
        messages = load_conversation(thread_id)
        title = "New Chat"
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # Truncate the first message to 25 characters for a clean sidebar UI
                title = msg.content[:25] + "..." if len(msg.content) > 25 else msg.content
                break
        st.session_state['thread_titles'][thread_id] = title

    button_label = st.session_state['thread_titles'][thread_id]

    # Render the button with the dynamic title, but use the UUID as the unique key
    if st.sidebar.button(button_label, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:  
            if isinstance(message, HumanMessage):
                temp_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage) and message.content:
                temp_messages.append({"role": "assistant", "content": message.content})

        st.session_state['message_history'] = temp_messages

#************************* MAIN UI**********************************

for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.write(message["content"]) 

user_input = st.chat_input("Enter your message:")

if user_input:
    st.session_state['message_history'].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)
    
    CONFIG = {
        "configurable": {"thread_id": st.session_state['thread_id']},
        "metadata": {
            "thread_id": st.session_state['thread_id']
        },
        "run_name": "chat_turn"
    }
    
    with st.chat_message("assistant"):
        status = st.status("Agent thinking...", expanded=True)

        def ai_only_stream(status_container):
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if message_chunk.__class__.__name__ == "AIMessageChunk":
                    if hasattr(message_chunk, "tool_call_chunks") and message_chunk.tool_call_chunks:
                        for chunk in message_chunk.tool_call_chunks:
                            if "name" in chunk and chunk.get("name"):
                                status_container.write(f"🛠️ Calling tool: **{chunk['name']}**...")

                    content = message_chunk.content
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                yield item["text"]
                    elif isinstance(content, str):
                        yield content
                
                elif message_chunk.__class__.__name__ in ["ToolMessage", "ToolMessageChunk"]:
                    tool_name = message_chunk.name if hasattr(message_chunk, 'name') else 'Tool'
                    status_container.write(f"✅ Finished: **{tool_name}**")
                    
                    if tool_name == "rag_tool":
                        with status_container.expander("📄 View Retrieved Document Context"):
                            st.write(message_chunk.content)

        ai_message = st.write_stream(ai_only_stream(status))
        status.update(label="Response generated!", state="complete", expanded=False)
        st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
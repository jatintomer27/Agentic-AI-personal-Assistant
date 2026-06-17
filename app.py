"""
Streamlit frontend for the Agentic AI Chatbot.
"""


# --------------------------------------------
# 
# 1. Importing the Libraries
# 
# --------------------------------------------

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from chatbot import logger
from chatbot.rag.pipeline import RAGPipeline
from chatbot.utils.common import (
    generate_session_id,
    load_config_file,
    load_conversation
)
from chatbot.database import (
    init_db,
    save_session_name,
    get_all_sessions
)

# --------------------------------------------
# 
# 2. Load the Configuration
# 
# --------------------------------------------




try:
    config = load_config_file(__file__)
except Exception as e:
    st.error(
            f"Failed to load Configuration file:\n{e}\n"
            f"Check logs for details."
        )
    st.stop()



# --------------------------------------------
# 
# 2. Initialize once at app startup
# 
# --------------------------------------------

@st.cache_resource(show_spinner=False)                  # runs only ONCE, not on every rerender
def get_db():
    """Init DB and return session factory."""
    try:
        SessionLocal = init_db()        # creates DB + tables if missing
        db_session = SessionLocal()
        return db_session
    except ValueError as e:
        st.error(
            f"⚠️ Database Configuration Error:\n{e}"
        )
        return None
    except Exception as e:
        st.error(
            f"⚠️ Database Initialization failed.\n\n"
            f"Check the Credentials of the Database in environment file."
        )
        return None

@st.cache_resource(show_spinner=False)
def get_chatbot():
    """Build the LangGraph chatbot once at startup."""
    try:
        from chatbot.graph.pipeline import get_chatbot as _get_chatbot
        return _get_chatbot()
    except FileNotFoundError as e:
        st.error(f"⚠️ Config file not found:\n{e}")
        return None
    except Exception as e:
        st.error(
            f"⚠️ Chatbot graph failed to initialize:\n\n{e}\n\n"
            f"Most likely cause: PostgreSQL is not running.\n"
            f"Check logs for details."
        )
        return None 

@st.cache_resource(show_spinner=False)
def get_rag():
    """Initialize RAG pipeline once at startup."""
    try:
        rag = RAGPipeline(config)
        rag.ingest_existing_files()   # process any already-uploaded files
        return rag
    except ValueError as e:
        st.error(
            f"⚠️ RAG Configuration Error:\n\n{e}"
        )
        return None
    except Exception as e:
        st.error(
            f"⚠️ RAG failed to start:\n\n{e}\n\n"
            f"Check logs for details."
        )
        return None

# DB Session
db_session = get_db()
if db_session is None:
    st.stop()
chatbot = get_chatbot()
if chatbot is None:
    st.stop()


# --------------------------------------------
# 
# 3. Initialized RAG pipeline
# 
# --------------------------------------------


if "rag_initialized" not in st.session_state:
    with st.status("🔄 Initializing RAG...", expanded=True) as status:
        rag = get_rag()
        if rag:
            status.update(label="✅ RAG Ready!", state="complete", expanded=False)
            st.session_state["rag_initialized"] = True
            st.session_state["rag"] = rag
        else:
            st.warning("⚠️ RAG unavailable — chatbot will answer from LLM knowledge only.")
else:
    rag = st.session_state["rag"]
    

# --------------------------------------------
# 
# 3. Utility Functions
# 
# --------------------------------------------

def reset_chat():
    session_id = generate_session_id()
    st.session_state['session_id'] = session_id
    st.session_state['message_history'] = []
    reload_chat_sessions()
    add_temp_session(session_id)

def add_temp_session(session_id):
    sessions = st.session_state['chat_sessions']
    session_name = sessions.get(session_id) or "New Chat"
    if session_id not in sessions:
        sessions = {session_id:session_name,**sessions} # add to top
        st.session_state['chat_sessions'] = sessions  

def reload_chat_sessions():
    st.session_state['chat_sessions'] = {
        s['session_id']: s['session_name']
        for s in get_all_sessions(db_session)
    }
def ai_only_stream(config, user_input):
    try:
        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            if metadata.get('langgraph_node') != 'llm_call':
                continue
            if isinstance(message_chunk, AIMessageChunk | AIMessage):
                content = message_chunk.content
                if isinstance(content, str):
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            yield block.get("text", "")
    except Exception as e:
        logger.exception(f"Exception occured while invoking the LLM: {e}")
        yield "Something went wrong check logs for more details"


# --------------------------------------------
# 
# 4. Session Initilizations
# 
# --------------------------------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = generate_session_id()
if 'chat_sessions' not in st.session_state:
    try:
        reload_chat_sessions()
    except Exception as e:
        st.error(f"⚠️ Error occured while fetching the previous chats:\n{e}")


# --------------------------------------------
# 
# 5. Config Setup
# 
# --------------------------------------------

SESSION_ID = st.session_state['session_id']
CONFIG = {
    'configurable':{
        'thread_id':SESSION_ID,
        'user_id':'user_1',
    },
}


# --------------------------------------------
# 
# 5. Sidebar UI
# 
# --------------------------------------------


st.sidebar.title('LangGraph Chatbot')

# ------------------- New Chat -------------------
if st.sidebar.button('New Chat', key='new_chat_btn'):
    reset_chat()
# ------------------- Documents Upload -------------------
st.sidebar.divider()
st.sidebar.header("📄 Documents")
uploaded_files = st.sidebar.file_uploader(
    label       = "Upload PDF or TXT",
    type        = ["pdf", "txt"],
    accept_multiple_files = True,
    key         = "file_uploader",
    help        = "Upload one or more PDF files to process."
)
# Process each uploaded file
if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            result = rag.ingest_file(uploaded_file)
        if result["status"] == "success":
            if result["already_existed"]:
                st.sidebar.info(f"'{uploaded_file.name}' already indexed")
            else:
                st.sidebar.success(
                    f"✅ '{uploaded_file.name}' indexed ({result['chunks']} chunks)"
                )
        else:
            st.sidebar.error(f"❌ Error: {result['error']}")
st.sidebar.subheader("📚 Indexed Files")
indexed_files = rag.get_indexed_files()
if indexed_files:
    for filename in indexed_files:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.text(filename)
        # delete button for each file
        if col2.button("🗑", key=f"del_{filename}"):
            file_deleted = rag.delete_file(filename)
            if file_deleted:
                st.rerun()
            else:
                st.error(f"Failed to delete the file, Check logs for more details.")
            
else:
    st.sidebar.caption("No files indexed yet")
# ------------------- My Conversations -------------------
st.sidebar.divider()
st.sidebar.header('My Conversations')
for session_id, session_name in st.session_state['chat_sessions'].items():
    if st.sidebar.button(str(session_name), key=session_id):
        st.session_state["session_id"] = session_id
        conversation = load_conversation(chatbot, session_id)
        st.session_state['message_history'] = conversation
        reload_chat_sessions()
        st.rerun()  # added this so UI updates immediately on click


# --------------------------------------------
# 
# 6. Main UI
# 
# --------------------------------------------


# Loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
# Take the user input
user_input = st.chat_input("Type Here")
if user_input:
    try:
        save_session_name(db_session, st.session_state.get('session_id'), user_input)
    except Exception as e:
        st.error(f"⚠️ Error occured while saving the session:\n{e}")
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    # Store the message in message_history and show on frontend
    with st.chat_message('assistant'):
        ai_message = st.write_stream(ai_only_stream(CONFIG,user_input))
        st.session_state['message_history'].append({'role':'assistant','content':ai_message})
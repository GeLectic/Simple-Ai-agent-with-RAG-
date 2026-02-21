from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

model = ChatGroq(
    model= "openai/gpt-oss-120b",
    temperature=0.0
)

# Global variables to dynamically hold the RAG components
vector_store = None
retriever = None

def ingest_document(file_path: str):
    """Reads a PDF, chunks it, and initializes or updates the vector store."""
    global vector_store, retriever
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embedding)
    else:
        # Add new documents to the existing vector store
        vector_store.add_documents(chunks)
        
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return True


# -----------------
# Tools
#------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the uploaded PDF document.
    Use this tool when the user asks factual / conceptual questions 
    that might be answered from the stored documents.
    """
    global retriever
    # Safety check if user asks a question before uploading a PDF
    if retriever is None:
        return "Error: No document has been uploaded yet. Ask the user to upload a document first using the sidebar."

    results = retriever.invoke(query)

    # Format the retrieved documents cleanly for the LLM
    formatted_context = ""
    for i, doc in enumerate(results):
        formatted_context += f"--- Document {i+1} ---\n{doc.page_content}\n\n"

    return formatted_context


@tool 
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol(e.g 'AAPL', 'TSLA')
    using Alpha Vantage with api key in the url 
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=WLIOJMOTESFP393E"
    response = requests.get(url)
    return response.json()

@tool 
def calculator(first_num: str, second_num: str, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two number.
    Supported operations: add, subtract, multiply, divide
    """
    if operation == "add":
        result = float(first_num) + float(second_num)
    elif operation == "subtract":
        result = float(first_num) - float(second_num)
    elif operation == "multiply":
        result = float(first_num) * float(second_num)
    elif operation == "divide":
        if float(second_num) == 0:
            return {"error": "Division by zero is not allowed."}
        result = float(first_num) / float(second_num)
    else:
        return {"error": f"Unsupported operation '{operation}'. Supported operations are: add, subtract, multiply, divide."}
    return {"result": result}


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tool = model.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call"""
    messages = state['messages']
    
    # Enforce standard JSON output for strict tools compatibility
    if len(messages) > 0 and not isinstance(messages[0], SystemMessage):
        sys_msg = SystemMessage(content="You are a precise AI. Always use standard JSON for tool calls. NEVER output raw XML `<function>` tags.")
        messages = [sys_msg] + messages
        
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

#-----------------
# Check Pointer
#-----------------
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# graph
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint_tuple in checkpointer.list(config=None):
        thread_id = checkpoint_tuple.config['configurable']['thread_id']
        all_threads.add(thread_id)
    return list(all_threads)
# Simple-Ai-agent-with-RAG-

<img width="1793" height="842" alt="Screenshot 2026-02-21 214828" src="https://github.com/user-attachments/assets/9f8354b7-eba1-4ded-b68f-da6be71520b4" />

<img width="1837" height="805" alt="Screenshot 2026-02-21 215143" src="https://github.com/user-attachments/assets/4c0e8fa3-2133-4b1b-bae4-be8af3ea8ae1" />

<img width="1918" height="732" alt="image" src="https://github.com/user-attachments/assets/6816cf09-381d-41f6-9ec2-df2455a6722e" />
# 🦜🕸️ LangGraph RAG Agent with Streamlit

A stateful, multi-tool AI assistant built with LangGraph and Streamlit. This project features a conversational agent equipped with persistent memory, web search, real-time stock price fetching, a calculator, and a dynamic Retrieval-Augmented Generation (RAG) engine for chatting with uploaded PDFs.

## ✨ Features
* **Conversational AI**: Powered by Google's `gemini-2.5-flash` for fast, accurate reasoning.
* **Dynamic Document RAG**: Upload any PDF directly through the Streamlit UI to extract and chat with its contents, powered by HuggingFace embeddings and a FAISS vector store.
* **Persistent Memory**: An SQLite-backed checkpointer ensures your chat history and agent state are seamlessly saved across sessions.
* **Tool-Calling Agent**: 
  * 🌐 Web Search (DuckDuckGo)
  * 📈 Stock Market Data (AlphaVantage)
  * 🧮 Custom Math Calculator
  * 📄 Document Context Retriever
* **Interactive UI**: A modern Streamlit interface featuring token streaming, live tool execution statuses, collapsible document context viewers, and auto-generated sidebar chat titles.

## 🛠️ Prerequisites
* Python 3.9+
* API Keys for Google Gemini (and optionally AlphaVantage).
Install dependencies:


## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/langgraph-streamlit-agent.git](https://github.com/yourusername/langgraph-streamlit-agent.git)
   cd langgraph-streamlit-agent

2. **Install requirements**
pip install -r requirements.txt

3. **Set up Environment Variables:**
Create a .env file in the root directory and add your API keys:

4. **Code snippet**
GOOGLE_API_KEY=your_gemini_api_key_here

**Run the Application:**

Bash
streamlit run app.py

# 🤖 Groq RAG Q&A Chatbot

> Ask questions from any website using LangChain, Cohere embeddings, and Groq-powered LLaMA 3.1 LLM — via Retrieval-Augmented Generation (RAG)

---

## 🚀 Features

- 🔗 Enter any **website URL**
- 🧠 Uses **Cohere embeddings** for semantic understanding
- ⚡ Inference powered by **Groq’s LLaMA 3.1 model** (`llama-3.1-8b-instant`)
- 📚 Retrieval-Augmented Generation (RAG) using **LangChain**
- 🔍 Get highly relevant answers based on scraped content
- 🖥️ Simple and clean **Streamlit UI**

---

## 🏗️ Tech Stack

| Component       | Tool/Service                     |
|----------------|----------------------------------|
| UI             | Streamlit                        |
| LLM            | `llama-3.1-8b-instant` via Groq  |
| Embeddings     | Cohere (`embed-english-v3.0`)    |
| Vector Store   | FAISS                            |
| Framework      | LangChain                        |
| Scraping       | WebBaseLoader + BeautifulSoup    |
| Deployment     | Streamlit Cloud (or local)       |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/groq-rag-chatbot.git
cd groq-rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📄 requirements.txt

```txt
streamlit
langchain
langchain-community
langchain-groq
cohere
faiss-cpu
python-dotenv
beautifulsoup4
```

---

## 🧠 How It Works

1. User enters a website URL  
2. LangChain loads and processes the webpage content  
3. The content is split into smaller chunks  
4. Cohere generates embeddings for each chunk  
5. FAISS stores embeddings for similarity search  
6. When a question is asked, relevant chunks are retrieved  
7. Groq’s LLaMA 3.1 model generates the final answer using retrieved context  

---

## 🛠️ Future Improvements

- Support PDF and document uploads  
- Add conversational memory for follow-up questions  
- Multi-page website crawling  
- Display source references with answers  
- Persist vector database locally  

---

## ✍️ Author

**Shivam Kumar**  
B.Tech CSE  

- LinkedIn: https://www.linkedin.com/in/shivam-kumar-49b954251/  
- GitHub: https://github.com/shivamkumar4756  

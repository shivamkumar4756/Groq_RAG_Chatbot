import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader

import cohere


# Load API keys from .env file
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not COHERE_API_KEY or not GROQ_API_KEY:
    st.error("API keys are missing. Please check your .env file.")
    st.stop()


st.title("Groq RAG Chatbot")


# Initialize Groq LLM
# Using a currently supported model to avoid deprecation issues
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)


# Prompt template
# We explicitly tell the model to answer only from the provided context
prompt = ChatPromptTemplate.from_template(
"""
Answer the question only using the information from the context below.
If the answer is not present in the context, say that the information is not available.

<context>
{context}
</context>

Question: {input}
"""
)


# Cohere embedding setup
co = cohere.Client(COHERE_API_KEY)


class MyCohereEmbedder:
    # Converts multiple documents into embeddings
    def embed_documents(self, texts):
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings

    # Converts user query into embedding
    def embed_query(self, text):
        response = co.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        return response.embeddings[0]

    # Allows the class to be used directly as a callable
    def __call__(self, text):
        return self.embed_query(text)


# Function to create vector embeddings from a given URL
def vector_embeddings(user_url):

    # Rebuild vector store only if URL changes
    if "vectors" not in st.session_state or st.session_state.get("last_url") != user_url:

        st.session_state.last_url = user_url

        # Load website content
        loader = WebBaseLoader(user_url)
        documents = loader.load()

        # Split into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_documents = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = MyCohereEmbedder()

        # Store in FAISS vector database
        vectors = FAISS.from_documents(final_documents, embeddings)

        st.session_state.vectors = vectors


# User input fields
user_url = st.text_input("Enter website URL")
user_question = st.text_input("Ask a question based on the website content")


if st.button("Create Embeddings") and user_url:
    vector_embeddings(user_url)
    st.success("Vector database created successfully.")


# Handle question answering
if user_question and "vectors" in st.session_state:

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.process_time()

    try:
        response = retrieval_chain.invoke({"input": user_question})

        end_time = time.process_time()

        st.subheader("Answer")
        st.write(response["answer"])

        st.write(f"Response Time: {round(end_time - start_time, 2)} seconds")

        # Optional: show retrieved documents for transparency
        with st.expander("View Retrieved Documents"):
            for i, doc in enumerate(response["context"]):
                st.write(f"Document {i + 1}")
                st.write(doc.page_content)
                st.write("------")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")

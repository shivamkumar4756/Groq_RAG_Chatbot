import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader

from langchain_cohere import CohereEmbeddings


# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not COHERE_API_KEY or not GROQ_API_KEY:
    st.error("API keys are missing. Please check your .env file.")
    st.stop()


st.title("Groq RAG Chatbot")


# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)


# Prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the question only using the information from the context below.

If the answer is not present in the context, say:
"The information is not available in the provided context."

<context>
{context}
</context>

Question: {input}
"""
)


# Initialize Cohere Embeddings
embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-english-v3.0"
)


# Function to create vector embeddings
def vector_embeddings(user_url):

    # rebuild vectorstore only if URL changes
    if "vectors" not in st.session_state or st.session_state.get("last_url") != user_url:

        st.session_state.last_url = user_url

        with st.spinner("Loading website content..."):

            # Load webpage
            loader = WebBaseLoader(user_url)
            documents = loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            final_documents = text_splitter.split_documents(documents)

            # Create FAISS vector store
            vectors = FAISS.from_documents(final_documents, embeddings)

            st.session_state.vectors = vectors


# User Inputs
user_url = st.text_input("Enter website URL")
user_question = st.text_input("Ask a question based on the website content")


# Create embeddings button
if st.button("Create Embeddings") and user_url:
    vector_embeddings(user_url)
    st.success("Vector database created successfully.")


# Handle question answering
if user_question and "vectors" in st.session_state:

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 4}
    )

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.process_time()

    try:

        response = retrieval_chain.invoke({
            "input": user_question
        })

        end_time = time.process_time()

        st.subheader("Answer")
        st.write(response["answer"])

        st.write(f"Response Time: {round(end_time - start_time, 2)} seconds")


        # show retrieved docs
        with st.expander("View Retrieved Documents"):

            for i, doc in enumerate(response["context"]):

                st.write(f"Document {i+1}")
                st.write(doc.page_content)
                st.write("------")

    except Exception as e:

        st.error(f"Something went wrong: {str(e)}")
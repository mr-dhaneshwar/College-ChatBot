import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pandas as pd

# Load the Google API key from the .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI with the API key
from google.generativeai import configure
configure(api_key=api_key)

# Function to load CSV data
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    text = "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))
    return text

# Create a vector store with embeddings
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Create a QA chain
def create_qa_chain():
    prompt_template = """
    Answer the question in detail based on the provided context. If the context does not contain the answer, say "Answer not found in context."
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Streamlit application
def main():
    st.set_page_config("College Chatbot")
    st.title("College Chatbot")

    if 'vector_store' not in st.session_state:
        # Load CSV data and create vector store
        csv_file_path = "data.csv"  # Change this to your CSV file path
        text = load_csv_data(csv_file_path)
        vector_store = create_vector_store(text)
        st.session_state['vector_store'] = vector_store
    else:
        vector_store = st.session_state['vector_store']

    # Chatbot interaction
    user_question = st.chat_input("Ask a question about the college:")
    if user_question:
        st.write("**👤:** "+user_question)
        docs = vector_store.similarity_search(user_question, k=3)  # Retrieve top 3 relevant documents
        chain = create_qa_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("**🤖:** ", response["output_text"])

if __name__ == "__main__":
    main()

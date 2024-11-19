from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import os,shutil
from geminiAPI import*

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def dltfaiss():
    try:
        shutil.rmtree("faiss_index")
    except Exception as e:
        print(e)


def get_pdf_text(filepath):
    text = ""
    try:
        pdf_reader = PdfReader(filepath)
        for page in pdf_reader.pages:
            text += page.extract_text() #or ""  # Ensures that None is handled
    except Exception as e:
        print(f"An error occurred: {e}")
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # dltfaiss()
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """


    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):

    if os.path.exists("faiss_index"):

        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        
        # new_db = FAISS.load_local("faiss_index", embeddings)
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)  

        chain = get_conversational_chain()

        
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        print(response)
        # st.write("Reply: ", response["output_text"])
        return response["output_text"]
    
    else:
        return "You Did not provide any PDF, First upload the PDF"


# Streamlit application
def main():
    st.set_page_config(page_title="College Chatbot")
    st.title("College Chatbot")

    # Hardcoded JSON file path (replace with your file path)
    pdf_file_path = "TAE PDF.pdf"  # Change this to your JSON file path
    
    raw_text = get_pdf_text(pdf_file_path)
    # st.write(raw_text)
    # st.code(raw_text)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    # Chatbot interaction
    
    user_question = st.chat_input("Ask a question about the college:")
    if user_question:
        st.write("**👤:** "+str(user_question))
        A = user_input(user_question)
        st.write("**🤖:** ",A)

if __name__ == "__main__":
    main()

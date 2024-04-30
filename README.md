# College Chatbot with Streamlit

This project is a college chatbot built using Streamlit and LangChain with Google Generative AI embeddings. The chatbot answers questions about a college based on a CSV-based knowledge base.

## Project Overview

The chatbot provides users with a conversational interface to ask questions about various aspects of a college, such as admissions, courses, faculty, events, and campus facilities. It uses Google Generative AI embeddings for creating a vector store and utilizes a question-answering (QA) chain to respond to user queries.

## Prerequisites

- Python 3.7 or higher
- A Google API key with access to Google Generative AI services

### Obtaining a Google API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project.
3. Navigate to "APIs & Services" > "Credentials".
4. Click "Create Credentials" and select "API Key".
5. Copy the API key and add it to a `.env` file in your project directory:
   ```dotenv
   GOOGLE_API_KEY=your_api_key_here
   ```

## To install the packeges

run command: pip install -r requirements.txt

## to run the application

run command: streamlit run app.py

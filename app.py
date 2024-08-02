# app.py

import streamlit as st
from transformers import pipeline

# Load the pre-trained question answering pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Load the transcript from a file
@st.cache_data
def load_transcript(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Get answer from the QA model
def get_answer(question, context, qa_pipeline):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def main():
    # Load the QA model and transcript
    qa_pipeline = load_qa_pipeline()
    transcript_text = load_transcript("transcript.txt")

    # Set up the Streamlit interface
    st.title("Medical Chatbot")
    st.write("Ask questions about the patient's session with the doctor.")

    # Text input for the user's question
    question = st.text_input("Enter your question here:")

    if question:
        # Get the answer and display it
        answer = get_answer(question, transcript_text, qa_pipeline)
        st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()

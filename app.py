# app.py

import streamlit as st
from transformers import pipeline

# Load the pre-trained question answering pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load the transcript from a file
@st.cache_data
def load_transcript(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Function to get answer from the QA model
def get_answer(question, context, qa_pipeline):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Initialize session state variables for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

def main():
    # Load the QA model and transcript
    qa_pipeline = load_qa_pipeline()
    transcript_text = load_transcript("transcript.txt")

    # Set up the Streamlit chat-like interface
    st.title("Medical Chatbot")
    st.write("Ask questions about the patient's session with the doctor below.")

    # Text input for the user's question
    question = st.text_input("Your Question:", key="user_input")

    # If a question is entered
    if st.button("Send") and question:
        # Get the answer from the model
        answer = get_answer(question, transcript_text, qa_pipeline)
        
        # Update the conversation history
        st.session_state.conversation.append({"user": question, "bot": answer})
        st.session_state.user_input = ""  # Clear the input field

    # Display the conversation history
    for chat in st.session_state.conversation:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}\n")

if __name__ == "__main__":
    main()

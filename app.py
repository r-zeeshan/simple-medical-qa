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
    st.write("Interact with the chatbot by asking questions about the patient's session with the doctor.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Text input for the user's question
    question = st.text_input("You:", key="user_input")

    if st.button("Send"):
        if question:
            # Get the answer from the model
            answer = get_answer(question, transcript_text, qa_pipeline)

            # Update chat history
            st.session_state.chat_history.append({"user": question, "bot": answer})

            # Clear the input field
            st.session_state.user_input = ""

    # Display the chat history
    for entry in st.session_state.chat_history:
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**Bot:** {entry['bot']}")

if __name__ == "__main__":
    main()

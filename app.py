import streamlit as st
from transformers import pipeline

### Loading the pretrained language model
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")


@st.cache_data
def load_transcript(file_path):
    with open(file_path, "r") as file:
        return file.read()

def get_answer(question, context, qa_pipeline):
    """
    Retrieves the answer to a given question from a provided context using a question answering pipeline.
    Args:
        question (str): The question to be answered.
        context (str): The context in which the question is being asked.
        qa_pipeline (QuestionAnsweringPipeline): The question answering pipeline used to find the answer.
    Returns:
        str: The answer to the given question.
    """
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def main():
    qa_pipeline = load_qa_pipeline()
    transcript_text = load_transcript("transcript.txt")

    st.title("Medical Chatbot")
    st.write("Ask questions about the patient's session with the doctor.")

    ### Default questions list
    default_questions = [
        "What is the patient’s illness?",
        "What did the doctor diagnose?",
        "What medicine did the doctor mention?",
        "How long does the medicine take to show effects?",
        "What precautions did the doctor mention?",
        "What activities did the doctor recommend?"
    ]

    question = st.selectbox("Choose a question:", [""] + default_questions)
    user_question = st.text_input("Or enter your own question:")

    final_question = user_question if user_question else question

    if final_question:
        answer = get_answer(final_question, transcript_text, qa_pipeline)
        st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()

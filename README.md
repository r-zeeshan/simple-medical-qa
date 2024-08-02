
# Medical Chatbot Application

This Streamlit application allows users to interact with a medical transcript and get answers to common questions about a patient's session with a doctor. The app leverages the `distilbert-base-uncased-distilled-squad` model from Hugging Face for question answering.

## Features

- Ask predefined questions about the medical session.
- Input custom questions for specific inquiries.
- Receive answers based on the provided transcript.

## Usage

You can access the application online via the following link:

[**Medical Chatbot Application**](https://simple-mdc.streamlit.app/)

## Running Locally

To run the application locally, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/r-zeeshan/simple-medical-qa.git
   cd simple-medical-qa
   ```

2. **Install Dependencies:**

   Make sure you have Python installed, then install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

4. **Access the Application:**

   Open your web browser and go to `http://localhost:8501` to use the app.

## File Structure

- `app.py`: Main application script.
- `transcript.txt`: File containing the transcript used as context for the QA model.
- `requirements.txt`: List of Python dependencies.

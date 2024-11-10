import streamlit as st
from typing import Generator
from groq import Groq

st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Goes Brrrrrrrr...")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🏎️")

st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

st.write("## Welcome to the Model Interface!")
st.write("This app allows you to interact with **Model 1** for generating responses based on your inputs.")

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set the model directly (e.g., "llama3-70b-8192")
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3-70b-8192"  # Set your desired model directly here

# Define model details
models = {
    "llama3-70b-8192": {
        "name": "LLaMA3-70b-Instruct",
        "tokens": 8192,
        "developer": "Meta",
    },
    "llama3-8b-8192": {
        "name": "LLaMA3-8b-Instruct",
        "tokens": 8192,
        "developer": "Meta",
    },
    "mixtral-8x7b-32768": {
        "name": "Mixtral-8x7b-Instruct-v0.1",
        "tokens": 32768,
        "developer": "Mistral",
    },
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
}

# The selected model is already set, so no need for a dropdown
selected_model = st.session_state.selected_model  # This will always be the fixed model you've set

# Show the selected model
# st.write(f"Selected model: {models[selected_model]['name']}")

# Set max_tokens directly to the maximum value for the selected model
max_tokens = models[selected_model]["tokens"]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

import pandas as pd

CSV_FILE_PATH1 = 'patient_data_with_multiple_diagnoses.csv'
CSV_FILE_PATH2 = 'medicine_side_effects_single_row.csv'

def get_patient_diagnosis(patient_id):
    try:
        # Load the CSV containing patient data (make sure the CSV is structured properly)
        df = pd.read_csv(CSV_FILE_PATH1)
        
        # Filter by patient ID
        patient_data = df[df["patient_id"] == patient_id]
        
        if patient_data.empty:
            return None  # No diagnosis found for the patient ID

        # Sort the patient data by diagnosis_date and pick the latest
        patient_data["date"] = pd.to_datetime(patient_data["date"], errors='coerce')
        latest_diagnosis = patient_data.sort_values("date", ascending=False).loc[0]
        
        # Format the diagnosis to return the necessary context
        # diagnosis_context = f"Diagnosis for patient {patient_id} (latest): {latest_diagnosis['diagnosis']} on {latest_diagnosis['date'].strftime('%Y-%m-%d')}."
        medicine_leaflet = latest_diagnosis['medicine_leaflet']

        df = pd.read_csv(CSV_FILE_PATH2)
        side_efects_data = (df[df["medicine_leaflet"] == medicine_leaflet])["side_effects"].loc[0]

        # text_joint = medicine_leaflet + ': ' +side_efects_data

        return medicine_leaflet, side_efects_data
    
    except Exception as e:
        st.error(f"Error fetching diagnosis: {str(e)}")
        return None
    

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Get the prompt from the user
if prompt := st.chat_input("Enter your prompt here..."):
    # Add the user's message to the session history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user's message
    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(prompt)
    
    patient_id = prompt.split("Patient:")[1].split(",")[0].strip()  # Extract patient ID
    question = prompt.split("have")[1].strip().split(',')  # Extract the question
    symptoms = ', '.join(question)
    # print( f'My input: {patient_id} {question} {len(question)}' )

    medicine_leaflet, side_effects_data = get_patient_diagnosis(patient_id)
    # print(f"""Context: {medicine_leaflet}, side: {side_efects_data} \nThe patient said, I have {', '.join(question)} """)

    context = """Instructions for the Model:

                I am giving you a large block of text that contains both drug information (medicine leaflet) and a patient's symptoms. Based on this information, analyze whether the symptoms described are typical side effects of the drug or if they indicate something unusual that may require medical attention.

                Big Text Block:
                {medicine_leaflet}

                Side effects:
                {side_effects_data}

                Patient Query:

                The patient said, "I have {symptoms} Should I be worried?"

                Model's Task:

                Analyze the Symptoms: Check if {symptoms} are listed as common side effects in the leaflet.
                Provide a Short Answer:
                - If common: State that these symptoms are **NORMAL** side effects of {medicine_leaflet}, but suggest monitoring and consulting a doctor if symptoms persist.
                - If uncommon: State that these symptoms are not common, advise **STOPPING** the medication, and suggest seeking medical help immediately.
                """

    context_filled = context.format(medicine_leaflet=medicine_leaflet, 
                                side_effects_data=side_effects_data, 
                                symptoms=symptoms)
    print( f"""Context: {context_filled}""" )

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=selected_model,  # Use the fixed selected model here
            messages=[{
                "role": "user",  # Still specifying role as "user" for the new message
                "content": context_filled  # Pass the context string as content
            }],
            max_tokens=max_tokens,  # Use the pre-set max_tokens value
            stream=True,
        )

        # Use the generator function with st.write_stream to display the response in real-time
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)

        # Append the assistant's full response to session_state.messages
        if isinstance(full_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response}
            )

    except Exception as e:
        st.error(e, icon="🚨")

# Debugging output (you can remove this in production)
print(f'Full messages: {st.session_state.messages}')
print('--' * 10)
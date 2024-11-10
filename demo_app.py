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

icon("💊")

st.subheader("LlamaMed Guardian: Medication Safety Assistant", divider="rainbow", anchor=False)

st.write("## Welcome!")
st.write("This app allows you search on historical record of patients to support you.")

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

CSV_FILE_PATH1 = 'EVOLUCAO.csv'
CSV_FILE_PATH2 = 'leaflet_side_effects.csv'
CSV_FILE_PATH3 = 'summary_patients.csv'

def get_patient_diagnosis(patient_id):
    try:
        # Load the CSV containing patient data (make sure the CSV is structured properly)
        df = pd.read_csv(CSV_FILE_PATH1, sep='~')
        print(df.head())
        print(patient_id, df.dtypes)
        
        # Filter by patient ID
        patient_data = df[df["Admission_ID"].astype(str) == str(patient_id)]
        # print( patient_data.shape[0] )
        
        if patient_data.empty:
            return None, None, None  # No diagnosis found for the patient ID

        # Sort the patient data by diagnosis_date and pick the latest
        patient_data["date"] = pd.to_datetime(patient_data["Note_Date"], errors='coerce')
        # latest_diagnosis = patient_data.sort_values("date", ascending=False).loc[0]
        
        # Format the diagnosis to return the necessary context
        # diagnosis_context = f"Diagnosis for patient {patient_id} (latest): {latest_diagnosis['diagnosis']} on {latest_diagnosis['date'].strftime('%Y-%m-%d')}."
        # medicine_leaflet = latest_diagnosis['medicine_leaflet']

        df = pd.read_csv(CSV_FILE_PATH2, sep='~')
        side_efects_data = df["summary_en"].loc[0]
        print(side_efects_data)

        df = pd.read_csv(CSV_FILE_PATH3)
        patients_summary = df[df["patient_id"].astype(str) == str(patient_id)]
        patient_summary = patients_summary['summary'].loc[0]
        
        # text_joint = medicine_leaflet + ': ' +side_efects_data

        return '', side_efects_data, patient_summary
    
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

    # Check if the input matches the expected pattern
    if "Patient:" not in prompt or "the patient said:" not in prompt:
        # If the input doesn't match the expected pattern, show an error message
        st.chat_message("assistant", avatar="🤖").markdown("Please follow the pattern: `Patient: <patient_id>, the patient said: I have <symptoms>`.")
    else:
        try:
            # Extract patient_id and symptoms from the input
            patient_id = prompt.split("Patient:")[1].split(",")[0].strip()  # Extract patient ID
            question = prompt.split("the patient said:")[1].strip()  # Extract the question
            symptoms = question.split(",")  # Extract symptoms
            print( f"Patient {patient_id}" )

            # Retrieve patient diagnosis and other info
            medicine_leaflet, side_effects_data, patient_summary = get_patient_diagnosis(patient_id)

            # If patient data is not found, show the "user does not exist" message
            if medicine_leaflet is None:
                st.chat_message("assistant", avatar="🤖").markdown("User does not exist.")
            else:
                # Fill the context template if patient is found
                context = """Instructions for the Model:

                            You are given a block of text containing both drug information (medicine leaflet) and a patient’s reported symptoms, along with additional details about the patient’s history. Based on this information, your task is to analyze the symptoms described and determine whether they are typical side effects of the medication or if they suggest something unusual that may require immediate medical attention.

                            Big Text Block:
                            {medicine_leaflet}

                            Side effects:
                            {side_effects_data}

                            Patient Query:

                            The patient said, "I have {symptoms} Should I be worried?"

                            Additional Patient Summary:
                            {patient_summary}

                            Model's Task:

                            Analyze the Symptoms: Check if {symptoms} are listed as common side effects in the leaflet.
                            Consider Patient’s History: Take into account the patient's existing conditions (e.g., hepatic steatosis) and medication regimen.

                            Provide a Short Answer:
                            - If common: State that these symptoms are **NORMAL** side effects of {medicine_leaflet}, but suggest monitoring and consulting a doctor if symptoms persist.
                            - If uncommon: State that these symptoms are not common, advise **STOPPING** the medication, and suggest seeking medical help immediately.
                            """

                context_filled = context.format(medicine_leaflet=medicine_leaflet, 
                                                side_effects_data=side_effects_data, 
                                                symptoms=", ".join(symptoms), patient_summary=patient_summary)
                print(f"""Context: {context_filled}""")

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

        except Exception as e:
            st.error(f"An error occurred: {e}", icon="🚨")


# Debugging output (you can remove this in production)
print(f'Full messages: {st.session_state.messages}')
print('--' * 10)
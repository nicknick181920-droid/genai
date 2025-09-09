# This script is designed to be run in a Google Colab environment.
# Before running, make sure to install the required libraries by opening a cell and running:
# !pip install -r requirements.txt

# You will need to restart the runtime after installation.

import gradio as gr
import torch
from transformers import pipeline

# --- 1. Model and Pipeline Setup ---
# Initialize the text-generation pipeline with the specified model.
# Using the "pipeline" abstraction from transformers is a high-level helper
# that handles a lot of the boilerplate for you.
print("Loading the IBM Granite model. This may take a few minutes...")
try:
    # Use the pipeline to simplify model and tokenizer loading
    pipe = pipeline("text-generation", model="ibm-granite/granite-3.1-2b-base", device_map="auto")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None # Set pipe to None if loading fails to handle errors gracefully

# --- 2. Chatbot Logic ---
def get_chatbot_response(prompt: str, user_demographic: str) -> str:
    """
    Generates a financial advice response using the loaded model.
    The response is tailored based on the user's demographic.
    
    Args:
        prompt (str): The user's question or query.
        user_demographic (str): The selected demographic ("Student" or "Professional").
    
    Returns:
        str: The chatbot's generated response.
    """
    if not pipe:
        return "Sorry, the model failed to load. Please check the setup and try again."

    # Craft a system prompt to guide the model's behavior.
    # This acts as the "persona" for the chatbot.
    # We are asking it to be a financial advisor and to tailor its advice.
    system_prompt = f"""You are a helpful and knowledgeable financial advisor.
    You provide concise, actionable, and personalized financial guidance.
    Your advice on savings, taxes, and investments is tailored to the user's demographic.

    User Demographic: {user_demographic}
    """
    
    # Combine the system prompt and the user's input for the model.
    # The model will use this full context to generate its response.
    full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    
    try:
        # Generate the text using the model.
        # max_new_tokens controls the length of the generated response.
        # num_return_sequences specifies how many different responses to generate (we just need one).
        response = pipe(full_prompt, max_new_tokens=256, num_return_sequences=1)
        
        # Extract and format the generated text.
        # We clean up the response by removing the original prompt and
        # stripping any leading/trailing whitespace.
        generated_text = response[0]['generated_text']
        cleaned_text = generated_text.replace(full_prompt, "").strip()
        
        return cleaned_text
    
    except Exception as e:
        print(f"Error during text generation: {e}")
        return "An error occurred while generating the response. Please try again."

def chat_with_persona(user_message, history, user_demographic):
    """
    A wrapper function for the Gradio ChatInterface.
    This function handles the history and passes the user_demographic
    to the core generation function.
    
    Args:
        user_message (str): The user's input from the Gradio interface.
        history (list): The conversation history.
        user_demographic (str): The selected demographic.
    
    Returns:
        tuple: An updated history and the generated response.
    """
    # The history is not used in this simplified example, but it's
    # included for future functionality.
    
    response = get_chatbot_response(user_message, user_demographic)
    
    # Gradio's ChatInterface expects a tuple (history, response)
    # where the history is a list of tuples (user_message, bot_response).
    history.append((user_message, response))
    return history, "" # Return an empty string for the next user input box

# --- 3. Gradio Interface Setup ---
# We define the input components for our chatbot.
demographic_dropdown = gr.Dropdown(
    label="Select your demographic",
    choices=["Student", "Professional"],
    value="Professional" # Default value
)

# Set up the Gradio ChatInterface.
# The ChatInterface automatically provides a text input box and a "Send" button.
# We pass our `chat_with_persona` function to the `fn` parameter.
if pipe:
    demo = gr.ChatInterface(
        fn=chat_with_persona,
        chatbot=gr.Chatbot(label="Personal Finance Chatbot"),
        textbox=gr.Textbox(placeholder="Ask me anything about finance...", container=False, scale=7),
        title="Hackathon: Personal Finance Chatbot",
        description="""
        Your personal financial guide. Ask about savings, taxes, or investments.
        Select your demographic to get tailored advice!
        """,
        additional_inputs=[demographic_dropdown],
        theme="soft"
    )

    # Launch the Gradio app. `share=True` generates a public link for Colab.
    demo.launch(share=True)
else:
    # If the model failed to load, display an error message
    print("Gradio interface will not launch due to model loading error.")
    print("Please check the `get_chatbot_response` function for error details.")

import os
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import speech_recognition as sr
import pyttsx3
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Load Groq API Key from environment variable (or set it directly)
os.environ["GROQ_API_KEY"] = "YOUR_GROQ-AI_API_KEY"

# Initialize the LLM with ChatGroq
llm = ChatGroq(model_name="llama3-8b-8192")

# Define memory for conversation retention
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a chat prompt template with explicit debt collection role
prompt = ChatPromptTemplate.from_template(
    """
    You are a professional debt collection assistant conducting a polite and structured call with a customer.
    Your goal is to remind the customer about their due payment, inquire about any delays, and obtain an expected payment date.
    Maintain an interactive telephonic conversation, responding to previous interactions accordingly.
    
    Customer Details:
    Name: {CustName}
    Phone: {Phone}
    Debt Amount: {DebtAmount}
    Due Date: {DueDate}
    
    Conversation Flow:
    1. Greet the customer and confirm if you are speaking with {CustName}.
    2. If confirmed, introduce yourself as a debt collection representative and clarify the intent of the call.
    3. Politely remind them of their due payment and due date.
    4. Ask if there is any reason for the delay in payment.
    5. If a reason is given, acknowledge it and ask for an expected payment date.
    6. Confirm the details and thank the customer before ending the call.
    
    Chat history:
    {chat_history}
    
    User: {user_input}
    AI:
    """
)

# Create the chatbot chain
chatbot_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Set speech rate to 150 words per minute

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            print("Sorry, there was an issue with the request.")
            return None

def save_to_excel(customer_data, file_name="debt_collection_data.xlsx"):
    df = pd.DataFrame([customer_data])
    if os.path.exists(file_name):
        existing_df = pd.read_excel(file_name)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_excel(file_name, index=False)
    print("Customer data saved successfully!")

def debt_collection_call(customer_info):
    print("Debt Collection AI is running. Say 'exit' to stop.")
    customer_info["Reason for Delay"] = ""
    customer_info["Expected Payment Date"] = ""
    
    # Step 1: Greet the customer and confirm their identity
    text_to_speech(f"Hello, am I speaking with {customer_info['CustName']}?")
    confirmation = speech_to_text()
    if confirmation and "yes" in confirmation.lower():
        
        # Step 2: Introduce the bot and clarify intent
        text_to_speech(f"Great! I am calling from the debt collection department regarding your due payment of {customer_info['DebtAmount']} which was due on {customer_info['DueDate']}.")
        
        # Step 3: Ask for the reason for delay
        text_to_speech("Can you please share the reason for the delay?")
        reason = speech_to_text()
        if reason:
            customer_info["Reason for Delay"] = reason
        
        # Step 4: Ask for expected payment date
        text_to_speech("When do you expect to make the payment?")
        expected_date = speech_to_text()
        if expected_date:
            customer_info["Expected Payment Date"] = expected_date
        
        # Step 5: Confirm details and close the call
        text_to_speech("Thank you for your time. We appreciate your response and will note down your expected payment date. Have a great day!")
        save_to_excel(customer_info)
    else:
        text_to_speech("I apologize for the mistake. Have a good day!")
    
    print("Conversation ended.")

if __name__ == "__main__":
    sample_customer = {
        "CustName": "John",
        "Phone": "1234567890",
        "DebtAmount": "$100.00",
        "DueDate": "3/20/2025",
    }
    debt_collection_call(sample_customer)

import os
import pandas as pd
import speech_recognition as sr
import pyttsx3
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load Groq API Key
os.environ["GROQ_API_KEY"] = "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"

# Initialize the LLM with fine-tuned parameters
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.2, top_p=0.8)

# Define memory for conversation retention
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="user_input")

# Define Chat Prompt Template with strict flow
prompt = ChatPromptTemplate.from_template(
    """
    You are NIA, a debt collection representative from ABC Financial Services. 
    Your goal is to professionally remind {CustName} about their due payment and obtain the reason for the delay with an expected payment date.
    **Follow this strict conversation flow and do not repeat information unnecessarily.**
    
    **Special Instructions:**
    - If the person says John is unavailable but will call back, simply acknowledge and end the call politely.
    - Do NOT repeat the payment details if they have already been stated and acknowledged.
    - Keep responses brief and professional.
    - If the user says "thank you," "thanks," or similar, assume they are closing the call and respond once before exiting.
    
    **Example Conversation:**
    **NIA:** Hello, am I speaking with {CustName}?
    **User:** Yes!
    
    **NIA:** Thank you, {CustName}. I’m calling from ABC Financial Services regarding your overdue payment of {DebtAmount}, which was due on {DueDate}. Can you share the reason for the delay?
    
    **User:** I had financial issues.
    **NIA:** I understand. When do you expect to make the payment?
    
    **User:** Next week.
    **NIA:** Thank you. I’ve noted that down. Have a good day.
    
    **User:** John will call you back.
    **NIA:** Understood. Please have John call us at his earliest convenience. Thank you for your time.
    
    **User:** Thank you.
    **NIA:** You're welcome. Have a great day.
    
    ----
    **Live Call:**
    **Customer Details:**
    - Name: {CustName}
    - Phone: {Phone}
    - Debt Amount: {DebtAmount}
    - Due Date: {DueDate}
    
    **Chat history:**
    {chat_history}
    
    **User:** {user_input}
    **NIA:**
    """
)

# Create the chatbot chain
chatbot_chain = prompt | llm | RunnablePassthrough()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def text_to_speech(text):
    """Converts AI response to speech, ensuring only relevant content is spoken."""
    clean_text = text.replace("NIA:", "").strip()
    print(f"NIA (spoken): {clean_text}")
    engine.say(clean_text)
    engine.runAndWait()

def speech_to_text():
    """Converts speech to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio).strip().lower()
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand.")
            return None
        except sr.RequestError:
            print("Error with recognition service.")
            return None

def debt_collection_call(customer_info):
    """Handles the debt collection call dynamically with AI."""
    print("Debt Collection AI is running. Say 'exit' or 'no thanks' to stop.")
    conversation_active = True
    first_turn = True
    user_response=""

    while conversation_active:
        # Load chat history
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        
        if first_turn:
            # AI initiates the conversation
            ai_response = chatbot_chain.invoke({
                "user_input": "",  # AI starts without waiting for user input
                "CustName": customer_info["CustName"],
                "Phone": customer_info["Phone"],
                "DebtAmount": customer_info["DebtAmount"],
                "DueDate": customer_info["DueDate"],
                "chat_history": chat_history
            })
            first_turn = False
        else:
            # Get user input
            user_response = speech_to_text()
            if user_response:
                if any(word in user_response for word in ["exit", "no thanks", "thank you", "thanks"]):
                    text_to_speech("You're welcome. Have a great day.")
                    conversation_active = False
                    break
                elif "call back" in user_response or "will call later" in user_response:
                    text_to_speech("Understood. Please have John call us at his earliest convenience. Thank you for your time.")
                    conversation_active = False
                    break
                
                # Get the AI's response
                ai_response = chatbot_chain.invoke({
                    "user_input": user_response,
                    "CustName": customer_info["CustName"],
                    "Phone": customer_info["Phone"],
                    "DebtAmount": customer_info["DebtAmount"],
                    "DueDate": customer_info["DueDate"],
                    "chat_history": chat_history
                })
        
        # Extract clean text from AIMessage
        ai_response_text = ai_response.content if hasattr(ai_response, "content") else str(ai_response)
        print(f"NIA: {ai_response_text}")
        text_to_speech(ai_response_text)

        # Save conversation memory
        memory.save_context(
            {"user_input": user_response if not first_turn else ""},
            {"NIA": ai_response_text}
        )
    print("Conversation ended.")

if __name__ == "__main__":
    sample_customer = {
        "CustName": "John",
        "Phone": "1234567890",
        "DebtAmount": "$100.00",
        "DueDate": "3/20/2025",
    }
    debt_collection_call(sample_customer)

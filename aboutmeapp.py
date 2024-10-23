#%%
import streamlit as st
import requests
import json
import logging
from typing import Optional
from ultralytics import YOLO
from PIL import Image

#%%

# Constants
BASE_API_URL = " https://5e3b-175-136-235-145.ngrok-free.app"
FLOW_ID = "c9617c76-b838-40e9-9944-d25656ec6908"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings
TWEAKS = {
  "ChatOutput-RYRQA": {},
  "Prompt-6Idet": {},
  "OpenAIModel-bVC7D": {},
  "TextInput-4VR8w": {},
  "Memory-fov4u": {},
  "Prompt-uqVIy": {},
  "ChatInput-6nMm3": {}
}


# Initialize logging
logging.basicConfig(level=logging.INFO)


# Function to run the flow
def run_flow(message: str,
             endpoint: str = FLOW_ID,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }

    if tweaks:
        payload["tweaks"] = tweaks

    headers = {"x-api-key": api_key} if api_key else None
    response = requests.post(api_url, json=payload, headers=headers)

    # Log the response for debugging
    logging.info(f"Response Status Code: {response.status_code}")
    logging.info(f"Response Text: {response.text}")

    try:
        return response.json()
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from the server response.")
        return {}


# Function to extract the assistant's message from the response
def extract_message(response: dict) -> str:
    try:
        # Extract the response message
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

# Function to run the flow
def main():
    st.title("Computer Vision + LLM 🤖")
    
    with st.sidebar: 
        # File uploader for image
        #uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        enable = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable)
        json_full = [] 
    
        if picture is not None:
            # Convert the uploaded file to an image
            image = Image.open(picture)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Perform inference on the uploaded image
            model= YOLO("yolo11n.pt")
            results = [] 
            if results is not None: 
                results = model(image)  # Use the uploaded image for inference                
                json_full = results[0].tojson()
            
                # Iterate over the results
                detected_classes = []
                for result in results:
                    detections = result.boxes
                    for box in detections:
                        class_id = int(box.cls[0])  # Class index
                        confidence = box.conf[0]  # Confidence score
                        class_name = model.names[class_id]  # Class name
                
                        detected_classes.append(class_name)
                        print(f"Detected: {class_name} with confidence: {confidence:.2f}")
            


    
    #text_input = st.text_input(detect)
    #st.write(json_full) #cross check detection result
    TWEAKS["TextInput-4VR8w"]["input_value"] = json_full
    

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages with avatars
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # Input box for user message
    if query := st.chat_input("Ask me anything..."):
        # Add user message to session state
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
                "avatar": "💬",  # Emoji for user
            }
        )
        with st.chat_message("user", avatar="💬"):  # Display user message
            st.write(query)

         # Call the Langflow API and get the assistant's response
        with st.chat_message("assistant", avatar="🤖"):  # Emoji for assistant
            message_placeholder = st.empty()  # Placeholder for assistant response
            with st.spinner("Thinking..."):
                # Fetch response from Langflow with updated TWEAKS and using `query`
                assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)

        # Add assistant response to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "avatar": "🤖",  # Emoji for assistant
            }
        )
        
        #if results is not None:         
        #[Optional] Show detection result
            #st.image(results[0].plot())

if __name__ == "__main__":
    main()
# %%

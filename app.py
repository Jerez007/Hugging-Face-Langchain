from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferWindowMemory
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")


# image to text model
def img2text(url):
    image_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text


# llm to generate short story
def generate_story(scenario):
    template = """
    You are telling a story to a friend.
    You can generate a short story based on a simple scenario. The story should be no more than 50 words and should use simple casual language as if you were telling a story to a friend.
    Do not make it sound like a poem or artistic. It should sound like the way a person tells another person a story for example while having coffee at a coffee shop.

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(
        llm=VertexAI(temperature=1),
        prompt=prompt,
        verbose=True,
    )

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story


# text to speech model to generate audio
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.mp3', 'wb') as file:
        file.write(response.content)

    # code below if for Elevenlabs. Not working yet. 
    # payload = {
    #     "model_id": "eleven_monolingual_v1",
    #     "text": message,
    #     "voice_settings": {
    #         "stability": 0,
    #         "similarity_boost": 0
    #     }
    # }

    # headers = {
    #     'accept': 'audio/mpeg',
    #     'xi-api-key': ELEVEN_LABS_API_KEY,
    #     'Content-Type': 'application/json'
    # }

    # response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/jBpfuIE2acCO8z3wKNLl?optimize_streaming_latency=0', json=payload, headers=headers)
    # if response.status_code == 200 and response.content:
    #     with open('audio.mp3', 'wb') as f:
    #         f.write(response.content)



def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="🚀")

    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded image.", use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.mp3")



if __name__ == '__main__':
    main()
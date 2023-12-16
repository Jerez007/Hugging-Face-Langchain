from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferWindowMemory

load_dotenv(find_dotenv())


# image to text model
def img2text(url):
    image_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    generate_story(text)
    return text

img2text("tokyo.webp")

# llm to generate short story
def generate_story(scenario):
    template = """
    You are a story teller:
    You can generate a short story based on a simple narrative. The story should be no more than 20 words;

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


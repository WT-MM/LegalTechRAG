import os
from dotenv import load_dotenv

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import chainlit as cl
from chainlit.input_widget import Slider

import google.generativeai as genai

from fastapi.responses import JSONResponse

from chainlit.auth import create_jwt
from chainlit.server import app



class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/embedding-001'
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]


def get_relevant_passages(query, db, n_results=10):
    passages = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    return passages


def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "")

    prompt = f"""question : {query}.\n
    Supplementary Information:\n {escaped}\n
    If you find that the question is unrelated to the additional information, you can ignore it and answer with 'OUT OF CONTEXT'.\n
     Your answer :
    """

    return prompt


def convert_pasages_to_string(passages):
    context = ""

    for passage in passages:
        context += passage + "\n"

    return context


config = {
    'max_output_tokens': 128,
    'temperature': 0.9,
    'top_p': 0.9,
    'top_k': 50,
}


@cl.on_chat_start
async def start():
    setUpGoogleAPI()
    loadVectorDataBase()

    settings = await cl.ChatSettings([
        Slider(
            id="temperature",
            label="Temperature",
            initial=config['temperature'],
            min=0,
            max=1,
            step=0.1,
        ),
        Slider(
            id="top_p",
            label="Top P",
            initial=config['top_p'],
            min=0,
            max=1,
            step=0.1,
        ),
        Slider(
            id="top_k",
            label="Top K",
            initial=config['top_k'],
            min=0,
            max=100,
            step=1,
        ),
        Slider(
            id="max_output_tokens",
            label="Max output tokens",
            initial=config['max_output_tokens'],
            min=0,
            max=1024,
            step=1,
        )
    ]).send()
    

    await setup_model(settings)

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    await cl.Message(content="Connected to Chainlit!").send()



@cl.on_settings_update
async def setup_model(settings):
    config['temperature'] = settings['temperature']
    config['top_p'] = float(settings['top_p'])
    config['top_k'] = int(settings['top_k'])
    config['max_output_tokens'] = int(settings['max_output_tokens'])
    
    model = genai.GenerativeModel(model_name="gemini-pro", generation_config=config)
    cl.user_session.set('model', model)
    

def setUpGoogleAPI():
    load_dotenv()

    api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=api_key)


def loadVectorDataBase():
    chroma_client = chromadb.PersistentClient(path="../database/")

    db = chroma_client.get_or_create_collection(
        name="sme_db", embedding_function=GeminiEmbeddingFunction())

    cl.user_session.set('db', db)

@app.get("/custom-auth")
async def custom_auth():
    # Verify the user's identity with custom logic.
    token = create_jwt(cl.User(identifier="Test User"))
    return JSONResponse({"token": token})
    
    
@cl.on_message
async def main(message):
    '''
    model = cl.user_session.get('model')
    
    question = message.content
    db = cl.user_session.get('db')
    passages = get_relevant_passages(question, db, 5)
    
    prompt = make_prompt(message.content, convert_pasages_to_string(passages))
    
    answer = model.generate_content(prompt)
    ansList = []
    for candidate in answer.candidates:
            ansList = [part.text for part in candidate.content.parts]

    ansRes = " ".join(ansList)
    
    await cl.Message(content=ansRes).send()
    '''
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    model = cl.user_session.get('model')
    
    question = message.content
    db = cl.user_session.get('db')
    passages = get_relevant_passages(question, db, 5)
    
    prompt = make_prompt(message.content, convert_pasages_to_string(passages))

    msg.content = " ".join([part.text for part in model.generate_content(prompt).candidates[0].content.parts])
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
    
    
    
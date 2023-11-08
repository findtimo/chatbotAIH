# source myenv/bin/activate
from io import BytesIO
import json
import logging
import tempfile
from typing import Final
from deep_translator import (GoogleTranslator, single_detection)
import requests
from telegram import Update, Message
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
import openai
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import CombinedMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from textblob import TextBlob
from dataclasses import dataclass

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_PROJECT"] = "Chatbot"

dotenv_path = os.path.join('./', 'local.env')
load_dotenv(dotenv_path)
TOKEN: Final = os.getenv("TELEGRAM_TOKEN")
BOT_USERNAME: Final = os.getenv("TELEGRAM_BOT")
DETECTLANG_KEY: Final = os.getenv("DETECTLANG_KEY")
IBM_API_KEY: Final = os.getenv("IBM_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
embedding = OpenAIEmbeddings()

# LLM Settings ----------------------------------------------------------------
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, max_tokens=80)
vectordb = Chroma(persist_directory="./vectordb", embedding_function=embedding)
# retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k: 2"})
retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k": 3})
print(vectordb._collection.count())

# Memory ----------------------------------------------------------------
# conv_memory = ConversationBufferWindowMemory(
#     memory_key="chat_history_lines",
#     input_key='question',
#     output_key='answer',
#     return_messages=True,
#     k=3
# )

# #Summary memory 
# summary_memory = ConversationSummaryMemory(llm=llm, input_key="question", memory_key="chat_history", output_key="answer")

# memory = CombinedMemory(memories=[conv_memory, summary_memory])

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key='question',
    output_key='answer'
)

#Create custom prompt
template = """
<SYSTEM> You are a dedicated assistant for Singapore HealthServe's volunteer onboarding.
<HISTORY> Conversation History:
{chat_history}
<CONTEXT> Your expertise is limited to matters related to HealthServe's volunteer onboarding, volunteering opportunities, and organisational information. Please do not respond to questions unrelated to HealthServe or volunteering:
{context}
If the question is not directly related to HealthServe or volunteering, simply ignore it, and let's continue focusing on HealthServe topics.
<USER> Question: {question}
<ASSISTANT> Answer (limited to 60 words):"""
custom_prompt = PromptTemplate.from_template(template)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True,
    memory=memory, 
    get_chat_history=lambda h : h
)


@dataclass
class Mood:
    emoji: str
    sentiment: float
    
def get_mood(input_text: str, *, threshold: float) -> Mood:
    sentiment: float = TextBlob(input_text).sentiment.polarity
    
    friendly_threshold: float = threshold
    hostile_threshold: float = -threshold
    
    if sentiment >= friendly_threshold:
        return Mood('😊', sentiment)
    elif sentiment <= hostile_threshold:
        return Mood('😡', sentiment)
    else:
        return Mood('😐', sentiment)
        

# Enable logging -------------------------------------------
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)



async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for chatting with me! I am here to answer your onboarding questions!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Ask me any questions!')

def handle_response(text: str ) -> str:
    text: str = text.lower()
    
    result = qa({"question": text})
    print("Response: \n",result["answer"])
    bot_reply = result["answer"]

    # Prints the source documents
    for i in range(len(result["source_documents"])):
        print("\nSource document", i, "\n", result["source_documents"][i])
    
    return bot_reply


# Create IAM Authenticator with your API key, Initialize the Speech to Text service
authenticator = IAMAuthenticator(IBM_API_KEY)
service = SpeechToTextV1(authenticator=authenticator)
service.set_service_url("https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/2aab828b-1e4b-4da7-aaf3-ab6c68382b58")

def download_and_prep(file_name: str, message: Message) -> None:
    message.voice.get_file().download(file_name)
    # message.reply_chat_action(action=ChatAction.TYPING)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    MSG_LIMIT_SEC = 1*60  # 1 minute (60 seconds)
    voice_message = update.message.voice

    if voice_message:
        if voice_message.duration > MSG_LIMIT_SEC:
            await update.message.reply_text("Voice message is too long (more than 1 minute). Please send a shorter voice message.", quote=True)
            return

    # Get the file ID and download the file using the bot object
    bot = context.bot
    file_id = voice_message.file_id
    file_content = await bot.get_file(file_id)
    binary_data = await file_content.download_as_bytearray()
    try:
        res = service.recognize(audio=binary_data, content_type='audio/ogg; codecs=opus', model='en-GB_Telephony').get_result()
        transcribed = res['results'][0]['alternatives'][0]['transcript']
        print("Transcribed: " , transcribed)
        confidence = res['results'][0]['alternatives'][0]['confidence']
        print("Confidence level: ", confidence)
        if(confidence < 0.7):
            await update.message.reply_text("Please say it again!")
        else:
            response: str = handle_response(transcribed)
            await update.message.reply_text(response)
    except Exception as e:
        print(f"An error occurred: {e}")
        update.message.reply_text("Could not get your voice message!")
        

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    
    to_translate: str = update.message.text
    # source_language = .detect(to_translate).lang
    lang = single_detection(to_translate, DETECTLANG_KEY)
    print("\nLanguag detected: "+lang)
    text = GoogleTranslator(source='auto', target='en').translate(to_translate)

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            text = text.replace(BOT_USERNAME, '').strip()
        else:
            return
        
    mood: Mood = get_mood(text, threshold=0.2) #Sensitivity of mood bot. Lower is sensitive
    print(f'{mood.emoji}: ({mood.sentiment})')
    if(mood.sentiment < -0.4):
        negativeResponse = "If you'd like to talk to someone and get the help you need, please consider reaching out to a helpline. They are available to provide assistance and guidance during challenging times.\n\n📞 Helpline: +65 3129 5000\n\nRemember, it's absolutely okay to seek help and take care of yourself. You're not alone, and there are resources available to support you. If you have any questions or need information about volunteering, please feel free to ask. We're here to assist you in any way we can."
        await update.message.reply_text(GoogleTranslator(source='en', target=lang).translate(negativeResponse))
    else:
        response: str = handle_response(text)
        print('Bot: ', response)
        await update.message.reply_text(GoogleTranslator(source='en', target=lang).translate(response))

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()

    #Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    #Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    
    #Speech
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    #Errors 
    app.add_error_handler(error)

    #Polls the Bot
    print('Polling...')
    app.run_polling(poll_interval=3)
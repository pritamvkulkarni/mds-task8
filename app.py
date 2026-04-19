from flask import Flask, request
import requests
import subprocess
import config

# Azure Speech
import azure.cognitiveservices.speech as speechsdk

# Azure Vision
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential

# Azure OpenAI
from openai import AzureOpenAI

app = Flask(__name__)

# GPT client
gpt_client = AzureOpenAI(
    api_key=config.AZURE_OPENAI_KEY,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01"
)

# Vision client
vision_client = ImageAnalysisClient(
    endpoint=config.AZURE_ENDPOINT,
    credential=AzureKeyCredential(config.AZURE_KEY)
)


def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})


def get_gpt_response(text):
    response = gpt_client.chat.completions.create(
        model=config.DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": text}]
    )
    return response.choices[0].message.content


def speech_to_text(file):
    speech_config = speechsdk.SpeechConfig(
        subscription=config.SPEECH_KEY,
        region=config.SPEECH_REGION
    )
    audio = speechsdk.AudioConfig(filename=file)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio
    )
    result = recognizer.recognize_once()
    return result.text


def analyze_image(file):
    with open(file, "rb") as f:
        result = vision_client.analyze(
            image_data=f,
            visual_features=["Caption", "Tags"]
        )
    caption = result.caption.text if result.caption else ""
    return caption


@app.route("/", methods=["POST"])
def webhook():
    data = request.json

    if "message" in data:
        msg = data["message"]
        chat_id = msg["chat"]["id"]

        # TEXT
        if "text" in msg:
            reply = get_gpt_response(msg["text"])

        # VOICE
        elif "voice" in msg:
            file_id = msg["voice"]["file_id"]
            file_info = requests.get(
                f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/getFile?file_id={file_id}"
            ).json()

            file_path = file_info["result"]["file_path"]
            url = f"https://api.telegram.org/file/bot{config.TELEGRAM_TOKEN}/{file_path}"

            open("voice.ogg", "wb").write(requests.get(url).content)

            subprocess.run(["ffmpeg", "-i", "voice.ogg", "voice.wav"])

            text = speech_to_text("voice.wav")
            reply = get_gpt_response(text)

        # IMAGE
        elif "photo" in msg:
            file_id = msg["photo"][-1]["file_id"]
            file_info = requests.get(
                f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/getFile?file_id={file_id}"
            ).json()

            file_path = file_info["result"]["file_path"]
            url = f"https://api.telegram.org/file/bot{config.TELEGRAM_TOKEN}/{file_path}"

            open("image.jpg", "wb").write(requests.get(url).content)

            desc = analyze_image("image.jpg")
            reply = get_gpt_response(desc)

        else:
            reply = "Send text, voice, or image."

        send_message(chat_id, reply)

    return "OK"
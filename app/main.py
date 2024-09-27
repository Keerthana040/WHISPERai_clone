from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import torchaudio
import os
from starlette.requests import Request
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

app = FastAPI()

# Mount static files like CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 template instance for rendering HTML
templates = Jinja2Templates(directory="templates")

# Declare global variables for the model and processor
model = None
processor = None


# Background task to preload the Whisper model asynchronously
@app.on_event("startup")
async def load_model():
    global model, processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")


def transcribe_audio(file_path: str) -> str:
    waveform, sample_rate = torchaudio.load(file_path)

    # Ensure the sample rate is 16000 Hz (Whisper's expected sample rate)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)

    # Generate transcription without gradients (since we're not training the model)
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def detect_language(file_path: str) -> str:
    waveform, sample_rate = torchaudio.load(file_path)

    # Ensure the sample rate is 16000 Hz (Whisper's expected sample rate)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)

    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])

    # Extract language token from the model's output
    language_token = processor.tokenizer.decode(generated_ids[0][:2])  # First two tokens
    return processor.tokenizer.convert_tokens_to_string(language_token)


def generate_word_cloud(text: str):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Save wordcloud image to an in-memory file
    img = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img, format="PNG")
    plt.close()
    img.seek(0)

    return img


@app.get("/", response_class=HTMLResponse)
def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe(request: Request, file: UploadFile = File(...)):
    global model
    if model is None:
        return HTMLResponse("Model is still loading, please wait...", status_code=503)

    file_path = f"uploads/{file.filename}"
    os.makedirs('uploads', exist_ok=True)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Perform the transcription
    transcription = transcribe_audio(file_path)

    return templates.TemplateResponse("result.html", {"request": request, "transcription": transcription})


@app.post("/language-detection", response_class=HTMLResponse)
async def detect_audio_language(request: Request, file: UploadFile = File(...)):
    if model is None:
        return HTMLResponse("Model is still loading, please wait...", status_code=503)

    file_path = f"uploads/{file.filename}"
    os.makedirs('uploads', exist_ok=True)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Perform language detection
    language = detect_language(file_path)

    return templates.TemplateResponse("result.html", {"request": request, "language": language})


@app.post("/word-cloud", response_class=StreamingResponse)
async def word_cloud(request: Request, file: UploadFile = File(...)):
    if model is None:
        return HTMLResponse("Model is still loading, please wait...", status_code=503)

    file_path = f"uploads/{file.filename}"
    os.makedirs('uploads', exist_ok=True)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Perform the transcription first
    transcription = transcribe_audio(file_path)

    # Generate word cloud from transcription
    img = generate_word_cloud(transcription)

    # Return the image as a StreamingResponse
    return StreamingResponse(img, media_type="image/png", headers={"Content-Disposition": "inline; filename=wordcloud.png"})

To install everything required to run your FastAPI application with the Whisper model, you'll need to follow these steps:

### Steps to Set Up the Project in PyCharm:

1. **Create a Virtual Environment**:
   - Open PyCharm.
   - Go to **File > Settings > Project: <Your Project> > Python Interpreter**.
   - Click on the gear icon and select **Add**.
   - Choose **New environment** and set it up with a virtual environment (venv).

2. **Install the Necessary Python Packages**:
   The Whisper model, FastAPI, and other libraries like `torchaudio`, `transformers`, and `wordcloud` need to be installed. Here is the list of dependencies and how to install them:

### Install the Core Dependencies:
```bash
pip install fastapi uvicorn[standard] torch torchaudio transformers wordcloud matplotlib jinja2
```

### Dependency Breakdown:
- **FastAPI**: Core framework for building the API.
- **Uvicorn**: ASGI server to run FastAPI applications.
- **Torch**: Required for the Whisper model, the base library for handling neural networks in PyTorch.
- **Torchaudio**: To handle audio processing, needed for Whisper.
- **Transformers**: Hugging Face's library for using pre-trained models like Whisper.
- **WordCloud**: To generate the word cloud from the transcription text.
- **Matplotlib**: For visualizing the word cloud image.
- **Jinja2**: For rendering HTML templates.

### Additional Libraries:
If you're going to use static files and CSS, you’ll also need to include **aiofiles** and **python-multipart** for handling file uploads:
```bash
pip install aiofiles python-multipart
```

### Download the Whisper Model:
After setting up the dependencies, the Whisper model will automatically be downloaded when you run the application for the first time. However, you can manually download it beforehand using the following command in Python:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Download the model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
```

This will ensure that the model files are cached locally.

### List of All Required Dependencies:
Here’s a full list of dependencies you need to install:
- `fastapi`
- `uvicorn`
- `torch`
- `torchaudio`
- `transformers`
- `wordcloud`
- `matplotlib`
- `jinja2`
- `aiofiles` (for serving static files)
- `python-multipart` (for handling file uploads)

### Run the FastAPI Server:
After installing the dependencies, you can run your FastAPI application with:
```bash
uvicorn main:app --reload
```
Make sure `main.py` is the name of your Python file where you define the FastAPI app.

### Additional Setup:
- **Static files**: Ensure your static files (CSS, etc.) are in the right directory (`static/`).
- **Templates**: Ensure your HTML templates are in the `templates/` folder.

### Optional: Create a `requirements.txt` file:
To make future installations easier, generate a `requirements.txt` file with the following command:
```bash
pip freeze > requirements.txt
```

This way, you or others can install all required packages using:
```bash
pip install -r requirements.txt
```
![WhatsApp Image 2025-05-21 at 22 41 50_2428d1c0](https://github.com/user-attachments/assets/40c95630-7fff-41d8-ae57-46c261280ebe)
![WhatsApp Image 2025-05-21 at 22 41 51_97932e5f](https://github.com/user-attachments/assets/735f73f1-a155-4abc-8cd9-aeb1fe776615)
![transcriber_1](https://github.com/user-attachments/assets/01ebdd2e-10de-419b-a701-771f264d5600)
![transcriber_2](https://github.com/user-attachments/assets/12850d58-30c6-49e7-85ae-032279a0bc00)




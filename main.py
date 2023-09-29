import io
import os
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, Form, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydub import AudioSegment

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE = "base"


model = WhisperModel(BASE, device="cpu", compute_type="int8", local_files_only=True)
model_lock = Lock()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/main", StaticFiles(directory="static", html=True), name="main")

@app.get("/")
async def root():
    return RedirectResponse("/main", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/message")
async def root():
    return {"message": "Hello user"}


@app.post("/audio1")
async def rec_file(
        la: Annotated[str, Form()],
        audio1: UploadFile = File(...)
):
    contents = await audio1.read()
    print(len(contents))
    s = io.BytesIO(contents)
    sound = AudioSegment.from_file(s)

    sound.export("f1.wav", format="wav")

    options_dict = {"language": la}
    options_dict["word_timestamps"] = True

    with model_lock:
        segments = []
        text = ""
        segment_generator, info = model.transcribe("f1.wav", beam_size=5, **options_dict)
        for segment in segment_generator:
            segments.append(segment)
            text = text + segment.text
        result = {
            "language": info.language,
            "segments": segments,
            "text": text
        }

    print(result)
    return {"result": result}


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE = "small"


model = WhisperModel("./model2", device="cpu", compute_type="float32", local_files_only=False)
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


prompt = ('А1 А2 А3 А4 А5 А6 А7 А8 А9 А10 ' +
'З1 З2 З3 З4 З5 З6 З7 З8 З9 З10 ' +
'Б1 Б2 Б3 Б4 Б5 Б6 Б7 Б8 Б9 Б10 ' +
'A Б В Г Д У Ж З И К Л М ')
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
    options_dict["word_timestamps"] = False
    options_dict["initial_prompt"] = prompt

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




# ct2-transformers-converter --model ./model --output_dir ./model2
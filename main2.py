import io
import os
import uuid
from typing import Annotated

import librosa
from fastapi import FastAPI, Form, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from transformers import WhisperForConditionalGeneration, WhisperProcessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

m1 = './model'
base = 'openai/whisper-base'

model = WhisperForConditionalGeneration.from_pretrained(m1)
processor = WhisperProcessor.from_pretrained(m1)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ru", task="transcribe")
prompt_text = (' ' +
               'А1 А2 А3 А4 А5 А6 А7 А8 А9 А10 ' +
               'З1 З2 З3 З4 З5 З6 З7 З8 З9 З10 ' +
               'Б1 Б2 Б3 Б4 Б5 Б6 Б7 Б8 Б9 Б10 ')
prompt_ids = processor.get_prompt_ids(prompt_text)

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
    speech, sr = librosa.load("f1.wav", sr=16000)

    input_features = processor(speech, return_tensors="pt", sampling_rate=16e3).input_features
    output_with_prompt = model.generate(input_features, language="ru", task="transcribe", prompt_ids=prompt_ids)
    # do_sample=True,
    # temperature=0.5,
    # top_p=0.5)

    strs = processor.batch_decode(output_with_prompt, skip_special_tokens=True)[0]
    # return_timestamps=True,
    # return_full_text=True)[0]

    print(strs)
    result = strs.replace(prompt_text, "")
    print(result)
    id = uuid.uuid1()
    sound.export('data/' + str(id) + ".wav", format="wav")
    with open('data/i.txt', 'a', encoding='utf-8') as file:
        file.writelines('\n' + str((str(id) + ' == ' + str(result))))
    return {"result": {"text": result}}


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

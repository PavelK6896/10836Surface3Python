FROM python:3.10.9-slim AS dependencies1
RUN python --version
RUN pip --version
ADD requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

FROM dependencies1 AS build

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


COPY main.py /main.py
COPY static/index.html /static/index.html
COPY .cache /root/.cache

RUN ln -s /root/.cache/huggingface/hub/models--guillaumekln--faster-whisper-base/blobs/867cf1a0fece1394e01d55e287ba2f09a577c046 /root/.cache/huggingface/hub/models--guillaumekln--faster-whisper-base/snapshots/515102184abb526d1cfb9c882107192588d7250a/config.json
RUN ln -s /root/.cache/huggingface/hub/models--guillaumekln--faster-whisper-base/blobs/d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9 /root/.cache/huggingface/hub/models--guillaumekln--faster-whisper-base/snapshots/515102184abb526d1cfb9c882107192588d7250a/model.bin
RUN ln -s /root/.cache/huggingface/hub/models--guillaumekln--faster-whisper-base/blobs/7818adb6de9fa3064d3ff81226fdd675be1f6344 /root/.cache/huggingface/hub/models--guillaumekln--faster-whisper-base/snapshots/515102184abb526d1cfb9c882107192588d7250a/tokenizer.json
RUN ln -s /root/.cache/huggingface/hub/models--guillaumekln--faster-whisper-base/blobs/c9074644d9d1205686f16d411564729461324b75 /root/.cache/huggingface/hub/models--guillaumekln--faster-whisper-base/snapshots/515102184abb526d1cfb9c882107192588d7250a/vocabulary.txt


WORKDIR /
ENV PORT 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT


# docker build --progress=plain -t s3-f7 -f f1.Dockerfile .
# docker run -e PORT=8000 -p 8000:8000 --name s3-f7c -d s3-f7

# docker login --username oauth --password secret cr.yandex
# docker image tag s3-f7 cr.yandex/crpbtkqol2ing4gt1s4p/stt1:v2
# docker push cr.yandex/crpbtkqol2ing4gt1s4p/stt1:v2


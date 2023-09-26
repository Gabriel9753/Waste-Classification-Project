FROM python:3.10-slim

WORKDIR /workspace
ADD requirements.txt /workspace/requirements.txt
RUN pip install -U pip && pip install -r /workspace/requirements.txt

COPY src /workspace/src
RUN python src/download_models.py

ENV HOME=/workspace
ENTRYPOINT python src/app.py
FROM python:3.10-slim

WORKDIR /workspace
ADD requirements.txt /workspace/requirements.txt
RUN pip install -U pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /workspace/requirements.txt

COPY src /workspace/src

ENV HOME=/workspace
ENTRYPOINT python src/app.py
FROM python:3.10-bookworm
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# update pip to support for whl.metadata -> less downloading
RUN pip install --no-cache-dir -U "pip>=24"

# create a working directory
RUN mkdir /root/foxy-whisp
WORKDIR /root/foxy-whisp
ENV PYTHONPATH=/root/foxy-whisp
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib"
# install the requirements for running the whisper server
RUN pip install --no-cache-dir faster-whisper librosa soundfile numpy torch torchaudio  tokenize_uk opus-fast-mosestokenizer==0.0.8.2
COPY silero_vad.py /root/foxy-whisp
COPY foxy-whisp.py /root/foxy-whisp
CMD ["python", "foxy-whisp.py"]

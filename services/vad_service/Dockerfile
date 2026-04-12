FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY vad_service/ vad_service/

# Pre-download Silero VAD model into the torch hub cache so the container
# works in air-gapped / restricted-network environments.
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=False, trust_repo=True)"

EXPOSE 8002

CMD ["uvicorn", "vad_service.app:app", "--host", "0.0.0.0", "--port", "8002"]

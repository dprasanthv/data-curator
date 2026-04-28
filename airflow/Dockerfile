FROM apache/airflow:2.9.3

USER airflow

# Heavy ML deps in their own layer so adding small libs below doesn't bust this cache.
RUN pip install --no-cache-dir \
    kafka-python-ng==2.2.3 \
    pyarrow==16.1.0 \
    torch==2.3.1 \
    torchvision==0.18.1 \
    transformers==4.41.2 \
    Pillow==10.3.0

# Pre-download CLIP weights (avoids macOS Docker volume cross-fs rename bug at runtime).
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); print('CLIP model cached')"

# Lightweight extras — adding/removing things here is cheap.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

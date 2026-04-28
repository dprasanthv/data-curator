FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small (~700 MB instead of 5 GB).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download both models so the app starts instantly.
RUN python -c "from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor; \
    AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
    AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    print('Models cached')"

COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

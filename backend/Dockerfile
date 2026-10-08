# CPU image for the WaxFashionStyleGAN API (Azure Container Apps).
# The StyleGAN2-ADA code (which needs Python < 3.12 for distutils) is baked in;
# the model weights are downloaded at startup with the HF_TOKEN secret.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN git clone --depth=1 https://github.com/researchpace/waxfashion.git /opt/waxfashion \
    && rm -rf /opt/waxfashion/.git

ENV STYLEGAN_DIR=/opt/waxfashion/stylegan2-ada-pytorch \
    HF_HOME=/tmp/hf \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

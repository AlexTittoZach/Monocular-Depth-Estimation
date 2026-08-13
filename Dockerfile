FROM python:3.12-slim

WORKDIR /app

# Copy the app's requirements
COPY Monocular-Depth-Estimation/requirements.txt ./requirements.txt

# Install dependencies with explicit limits
RUN pip install --no-cache-dir -r requirements.txt \
    "gradio==4.44.1" \
    "torch<=2.11.0" \
    "uvicorn>=0.14.0" \
    "websockets>=10.4" \
    "spaces==0.51.1"

# Copy the application code
COPY Monocular-Depth-Estimation ./Monocular-Depth-Estimation

# Set working directory to the app folder
WORKDIR /app/Monocular-Depth-Estimation

EXPOSE 7860

CMD ["python", "app.py"]

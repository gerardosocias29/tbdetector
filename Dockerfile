FROM python:3.9-slim

WORKDIR /app

# Set Streamlit and Matplotlib config directories to avoid permission issues
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV MPLCONFIGDIR=/tmp

# Install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files (avoids COPY . . problems)
COPY requirements.txt ./
COPY app.py .
COPY my_model.keras .

# Install Python packages
RUN pip3 install -r requirements.txt

# Create config to avoid CORS and port issues
RUN mkdir -p /tmp/.streamlit && \
    echo "[server]\nheadless = true\nport = 8501\nenableCORS = false\n" > /tmp/.streamlit/config.toml

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Use a lightweight official Python image
FROM python:3.11-slim

# Set work directory in the container
WORKDIR /app

# Install system dependencies (if gspread/Google auth ever need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Environment (optional defaults; you’ll override these in Railway)
# ENV ALPACA_BASE_URL=https://paper-api.alpaca.markets
# ENV KRAKEN_API_BASE_URL=https://api.kraken.com

# Default command – Railway cron can call this directly
CMD ["python", "main.py"]

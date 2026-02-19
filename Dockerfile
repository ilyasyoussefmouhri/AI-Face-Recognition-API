FROM python:3.13-slim

WORKDIR /src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user and fix permissions
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /src

# Set PYTHONPATH so Python can find the app module
ENV PYTHONPATH=/src

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
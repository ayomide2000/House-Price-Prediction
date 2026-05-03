FROM python:3.11-slim

WORKDIR /app

# Copy and install the Suitcase first (Layer Caching Habit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose FastAPI's default port
EXPOSE 8000

# Start the service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

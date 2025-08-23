# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Dependencies 
RUN pip install --no-cache-dir -r requirements.txt


# Expose FastAPI port
EXPOSE 8200

# Run FastAPI with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8200"]
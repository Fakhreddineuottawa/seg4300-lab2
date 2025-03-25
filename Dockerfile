# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . /app

# Install dependencies (including python-dotenv)
RUN pip install --no-cache-dir transformers torch flask python-dotenv

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project directory into the container
COPY . .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port number on which your FastAPI app runs
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main_db:app", "--host", "0.0.0.0", "--port", "8000"]

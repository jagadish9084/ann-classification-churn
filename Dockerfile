# Use official Python base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . /app

ENV TF_ENABLE_ONEDNN_OPTS=0

ENV CUDA_VISIBLE_DEVICES=""

# Install required dependencies
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run Flask application
CMD ["python", "app.py"]

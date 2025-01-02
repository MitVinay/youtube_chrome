FROM python:3.12.7-slim

WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential  # This will install gcc and other build tools

# Copy Flask app and other necessary files
COPY flask_app/ /app/

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Download NLTK stopwords and wordnet
RUN python -m nltk.downloader stopwords wordnet

# Expose port 8080
EXPOSE 8080

# Set the command to run the Flask app
CMD ["python", "app.py"]

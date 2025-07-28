FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for PyMuPDF and others
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with verbose output
RUN pip install --no-cache-dir -r requirements.txt --verbose > pip_install.log 2>&1 || { cat pip_install.log; exit 1; }

# Pre-download NLTK data
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt averaged_perceptron_tagger wordnet

# Pre-download T5 model
COPY setup_t5.py .
RUN python setup_t5.py > t5_download.log 2>&1 || { cat t5_download.log; exit 1; }

# Copy entire project folder
COPY . .

# Create output directory
RUN mkdir -p output

# Set volume for input/output
VOLUME /app

# Default command
CMD ["/bin/bash"]
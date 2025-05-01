FROM python:3.13

# Set working directory inside the container
WORKDIR /app

# Copy everything from your project directory into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (run downloader.py)
CMD ["python", "downloader.py"]
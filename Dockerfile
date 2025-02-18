# Use Python 3.9 base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the entire project structure
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run claude.py
CMD ["python", "claude/claude.py"]
# Use Python base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy files
COPY model.py .

# Install dependencies
RUN pip install scikit-learn

# Run script
CMD ["python", "model.py"]

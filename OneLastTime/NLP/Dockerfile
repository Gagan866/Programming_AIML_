FROM python:3.10-slim

# Set working directory
WORKDIR /app


# Copy everything to the container
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK dependencies
RUN python - <<EOF
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
EOF

# Run the script
CMD ["python", "model.py"]

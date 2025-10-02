FROM python

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .

COPY breast-cancer.csv .

CMD ["python", "model.py"]

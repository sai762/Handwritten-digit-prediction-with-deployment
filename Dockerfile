FROM python:3.8-slim

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD python app.py

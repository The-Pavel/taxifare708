FROM python:3.8.6-buster

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
RUN pip install /app

CMD uvicorn app.api:app --host 0.0.0.0
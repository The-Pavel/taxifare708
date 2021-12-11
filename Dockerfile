FROM python:3.8.6-buster

RUN mkdir app
COPY api.py /app
COPY model.joblib /app
COPY taxifare /taxifare
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn app.api:app --host 0.0.0.0 --port $PORT
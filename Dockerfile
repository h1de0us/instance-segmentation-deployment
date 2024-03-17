FROM python:3.7

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /app
WORKDIR /app

COPY src /app/src

ENTRYPOINT ["python3", "src/server.py"]
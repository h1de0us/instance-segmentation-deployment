FROM python:3.7

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app

COPY server.py /app
COPY grpc_server.py /app

COPY proto /app/proto
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/*.proto

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
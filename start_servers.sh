#!/bin/bash

python3 grpc_server.py &

uvicorn server:app --host 0.0.0.0 --port 8080
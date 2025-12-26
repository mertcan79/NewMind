# GRPC SERVICE PACKAGE INITIALIZATION
#
# TO COMPILE PROTOS, RUN:
#   python -m grpc_tools.protoc -I./protos --python_out=./grpc_service --grpc_python_out=./grpc_service ./protos/opinion_service.proto
#
# THEN FIX IMPORTS IN opinion_service_pb2_grpc.py:
#   Change: import opinion_service_pb2 as opinion__service__pb2
#   To: from grpc_service import opinion_service_pb2 as opinion__service__pb2

from .server import serve, OpinionAnalyzerServicer
from .client import OpinionAnalyzerClient

__all__ = ["serve", "OpinionAnalyzerServicer", "OpinionAnalyzerClient"]

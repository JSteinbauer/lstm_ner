# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import servicers.ner.ner_pb2 as ner__pb2


class NERStub(object):
    """///// Services ///////

    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ExtractEntities = channel.unary_unary(
                '/NER/ExtractEntities',
                request_serializer=ner__pb2.ExtractEntitiesRequest.SerializeToString,
                response_deserializer=ner__pb2.ExtractEntitiesResponse.FromString,
                )
        self.RunTraining = channel.unary_unary(
                '/NER/RunTraining',
                request_serializer=ner__pb2.RunTrainingRequest.SerializeToString,
                response_deserializer=ner__pb2.RunTrainingResponse.FromString,
                )


class NERServicer(object):
    """///// Services ///////

    """

    def ExtractEntities(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RunTraining(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_NERServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ExtractEntities': grpc.unary_unary_rpc_method_handler(
                    servicer.ExtractEntities,
                    request_deserializer=ner__pb2.ExtractEntitiesRequest.FromString,
                    response_serializer=ner__pb2.ExtractEntitiesResponse.SerializeToString,
            ),
            'RunTraining': grpc.unary_unary_rpc_method_handler(
                    servicer.RunTraining,
                    request_deserializer=ner__pb2.RunTrainingRequest.FromString,
                    response_serializer=ner__pb2.RunTrainingResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'NER', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class NER(object):
    """///// Services ///////

    """

    @staticmethod
    def ExtractEntities(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NER/ExtractEntities',
            ner__pb2.ExtractEntitiesRequest.SerializeToString,
            ner__pb2.ExtractEntitiesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RunTraining(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NER/RunTraining',
            ner__pb2.RunTrainingRequest.SerializeToString,
            ner__pb2.RunTrainingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
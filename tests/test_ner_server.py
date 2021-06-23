import subprocess
import time
from typing import List

import grpc
from servicers.ner import ner_pb2_grpc, ner_pb2
import pytest


class TestNerServer:

    @staticmethod
    @pytest.fixture(scope='function')
    def stub() -> ner_pb2_grpc.NERStub:
        """
        This fixture runs the server and yields a grpc client stub.
        After the test has been conducted, it terminates the server.
        """
        # Start the server and wait 5 sec until it's running
        server_subprocess = subprocess.Popen(['python', '-m', 'servicers.ner_server'], stdout=subprocess.PIPE)
        time.sleep(5)

        # Yield a client stub
        with grpc.insecure_channel("localhost:55555") as channel:
            yield ner_pb2_grpc.NERStub(channel)

        # Terminate the server.
        server_subprocess.terminate()

    @staticmethod
    @pytest.mark.parametrize('text, expected_entities', [
        ("hello this it Bruno from Jacobo", ['O', 'O', 'O', 'B-PER', 'O', 'B-LOC'])
    ])
    def test_ner_server(
            stub: ner_pb2_grpc.NERStub,
            text: str,
            expected_entities: List[str],
    ) -> None:
        """ Test running the server and sending an extract entities request. """
        response = stub.ExtractEntities(
            ner_pb2.ExtractEntitiesRequest(text=text)
        )
        # Check that the extracted entities are as expected.
        assert response.entities == expected_entities

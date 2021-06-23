from concurrent.futures import ThreadPoolExecutor
from typing import List

import grpc
import numpy as np

from ner.models.lstm_crf_model import NerLstmCrf
from servicers.config import DATA_DIR, MODEL_DIR, NER_PORT
from servicers.ner import ner_pb2_grpc
from servicers.ner.ner_pb2 import ExtractEntitiesResponse, RunTrainingResponse


class NerServer(ner_pb2_grpc.NERServicer):
    def __init__(self, data_dir: str, model_dir: str, load_weights: bool = True) -> None:
        """ Initialize the server with the data and model directories """
        data_dir: str = data_dir
        model_dir: str = model_dir
        self.ner_lstm_crf = NerLstmCrf(
            data_dir=data_dir,
            model_dir=model_dir,
        )
        if load_weights:
            self.ner_lstm_crf.load_weights(model_dir)

    def ExtractEntities(self, request, context) -> ExtractEntitiesResponse:
        """ Extract and return entities """
        text: str = request.text
        extracted_entities: List[str] = self.ner_lstm_crf.extract_entities(text)
        return ExtractEntitiesResponse(
            entities=extracted_entities
        )

    def RunTraining(self, request, context) -> RunTrainingResponse:
        """ Run training with specified number of epochs """
        self.ner_lstm_crf.train(epochs=request.epochs)

    def serve(self, max_workers: int = 10) -> None:
        """ Server the model with a maximal number max_workers concurrent threads """
        server = grpc.server(ThreadPoolExecutor(max_workers=max_workers))
        ner_pb2_grpc.add_NERServicer_to_server(self, server)
        server.add_insecure_port(f'[::]:{NER_PORT}')
        server.start()
        server.wait_for_termination()


if __name__ == '__main__':
    ner_server = NerServer(data_dir=DATA_DIR, model_dir=MODEL_DIR)
    ner_server.serve()



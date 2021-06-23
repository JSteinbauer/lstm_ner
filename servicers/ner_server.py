from ner.models.lstm_crf_model import NerLstmCrf
from servicers.ner import ner_pb2_grpc


class NerServer(ner_pb2_grpc.NERServicer):
    def __init__(self):
        data_dir: str = 'data/conll-2003_preprocessed/'
        model_dir: str = 'models/lstm_crf/'
        self.ner_lstm_crf = NerLstmCrf(
            data_dir=data_dir,
            model_dir=model_dir,
        )

    def ExtractEntities(self, request, context):

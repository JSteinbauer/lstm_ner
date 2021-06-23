import os

from dotenv import load_dotenv

# Load in environment variables
load_dotenv()
NER_PORT = int(os.getenv("NER_PORT", default="55555"))
DATA_DIR = os.getenv("DATA_DIR", default="data/conll-2003_preprocessed/")
MODEL_DIR = os.getenv("MODEL_DIR", default="model_training/lstm_crf/")
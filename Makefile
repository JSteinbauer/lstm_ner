NER_API_DIR := servicers/ner_api/
NER_SERVICER_DIR := servicers/ner/

download-glove:
	wget -P . "http://nlp.stanford.edu/data/glove.840B.300d.zip"
	unzip glove.840B.300d.zip -d glove.840B.300d.txt
	rm glove.840B.300d.zip

build:
	python build_vocab.py
	python build_glove.py

generate_grpc_protos:
	for f in $$(find ${NER_API_DIR} -name '*.proto'); do \
		python -m grpc_tools.protoc -I${NER_API_DIR} --python_out=${NER_SERVICER_DIR} --grpc_python_out=${NER_SERVICER_DIR} $$f; \
	done
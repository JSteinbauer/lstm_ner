NER_API_DIR := servicers/ner_api/
NER_SERVICER_DIR := servicers/ner/

# Download the glove embedding data
download-glove:
	wget -P . "http://nlp.stanford.edu/data/glove.840B.300d.zip"
	unzip glove.840B.300d.zip -d glove.840B.300d.txt
	rm glove.840B.300d.zip
	mv glove.* data/glove/

# Build the vocabulary and training-related glove array
build:
	python scripts/build_vocab.py
	python scripts/build_glove.py

# Generate servicers from grpc proto files
generate_grpc_servicers:
	for f in $$(find ${NER_API_DIR} -name '*.proto'); do \
		python -m grpc_tools.protoc -I${NER_API_DIR} --python_out=${NER_SERVICER_DIR} --grpc_python_out=${NER_SERVICER_DIR} $$f; \
	done

run_server:
	python -m servicers.ner_server
# Ubuntu Linux as the base image
FROM ubuntu:20.04

# Set UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install Python
RUN apt-get -y update && \
    apt-get -y upgrade

# The following line ensures that subsequent install doesn't expect user input
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3-pip python3-dev

# Install spaCy
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install gensim==3.8.3
RUN pip3 install wikipedia
RUN pip3 install spacy
RUN python3 -m spacy download en_core_web_trf
RUN python3 -m spacy download en_core_web_lg
RUN pip3 install stanza
RUN pip3 install sentencepiece
RUN pip3 install transformers
RUN pip3 install torch

# Add the files into container, under QA folder, modify this based on your need
RUN mkdir /QA
ADD ask /QA
ADD answer /QA
ADD modules.py /QA

# Change the permissions of programs
RUN chmod 777 /QA/ask
RUN chmod 777 /QA/answer

# Set working dir as /QA
WORKDIR /QA
ENTRYPOINT ["/bin/bash", "-c"]

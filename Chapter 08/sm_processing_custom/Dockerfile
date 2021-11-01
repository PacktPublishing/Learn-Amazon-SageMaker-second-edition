FROM python:3.7-slim

RUN pip3 install --no-cache gensim nltk sagemaker
RUN python3 -m nltk.downloader stopwords wordnet

ADD preprocessing-lda-ntm.py /

ENTRYPOINT ["python3", "/preprocessing-lda-ntm.py"]
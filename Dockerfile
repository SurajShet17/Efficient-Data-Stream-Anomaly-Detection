FROM python:3.9-slim-buster
ENV PYTHONIOENCODING='utf8'

WORKDIR /anomalib

COPY . /anomalib

WORKDIR /anomalib/anomalib

ENV HOME=/anomalib

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1


# Copy contents

RUN pip install -r requirements.txt


CMD ["python", "training.py"]


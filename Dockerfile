FROM python:3.9
RUN mkdir -m 777 /code
WORKDIR /code

COPY ./requirements.txt /install/requirements.txt
RUN pip install -r /install/requirements.txt

COPY . .

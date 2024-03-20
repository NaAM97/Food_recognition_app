FROM python:3.10.6

WORKDIR /app

RUN apt-get update
RUN apt-get install \
  'ffmpeg'\
  'libsm6'\
  'libxext6' -y

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD uvicorn API_NFG:app --host 0.0.0.0 --port 8000

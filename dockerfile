FROM python:3.10-slim

WORKDIR /python-docker

COPY requirements.txt requirements.txt

RUN apt-get update 

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y ffmpeg
 
COPY . .

EXPOSE 80

CMD [ "uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "80" ]
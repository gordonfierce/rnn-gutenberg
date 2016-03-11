FROM jupyter/scipy-notebook
EXPOSE 8000
WORKDIR /app
COPY . /app

VOLUME /app
VOLUME /tmp

RUN apt-get update -y; pip install -r requirements.txt
CMD hug -f app.py -p 5000

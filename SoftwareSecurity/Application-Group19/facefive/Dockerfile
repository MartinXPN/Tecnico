FROM python:3.7

## FOR DEBUG
# RUN apt install -yqq curl

# Flask listens by default on port 5000
EXPOSE 5000

WORKDIR /facefive

COPY requirements.txt /facefive
RUN pip3 install -r requirements.txt

COPY ./app/ /facefive/app
WORKDIR /

ENTRYPOINT ["gunicorn"]
CMD ["--bind", "0.0.0.0:5000", "--timeout", "10", "facefive.app:app"]
#CMD ["--certfile", "facefive/ssl.crt", "--keyfile", "facefive/ssl.key", "--bind", "0.0.0.0:5000", "facefive.app:app"]

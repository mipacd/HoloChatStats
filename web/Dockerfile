FROM python:3.13

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "gunicorn", "-k", "gevent", "-w", "5", "--bind", "0.0.0.0:80", "--timeout", "120", "server:app" ]
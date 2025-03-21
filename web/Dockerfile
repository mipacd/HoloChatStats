FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "gunicorn", "-w", "2","--bind", "0.0.0.0:80", "server:app" ]
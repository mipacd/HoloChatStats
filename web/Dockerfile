FROM python:3.13 as builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY . .
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 80

CMD [ "gunicorn", "-k", "eventlet", "-w", "5", "--bind", "0.0.0.0:80", "--timeout", "120", "server:app" ]
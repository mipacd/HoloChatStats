FROM python:3.13

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["sh", "-c", "python init_tool_store.py && python init_knowledge.py && uvicorn main:app --host 0.0.0.0 --port 8080"]
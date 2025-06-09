FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install django uvicorn

COPY . .

# Make sure the shared package is in Python path
ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
# lub jeśli chcesz ASGI:
# CMD ["uvicorn", "coronary_experiments.asgi:application", "--host", "0.0.0.0", "--port", "8000"]

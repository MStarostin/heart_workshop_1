FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install gunicorn==20.1.0

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "alfadrone.wsgi"]




FROM python:3.11-slim
# RUN apt-get update && apt-get install libgl1-mesa-glx -y
RUN apt-get update && apt-get install -y python3-opencv
COPY requirements.txt /app/requirements.txt
COPY src /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
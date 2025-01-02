FROM python:3.10
WORKDIR /usr/local/fastapi_app

# Install the application dependencies
COPY fastapi_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY fastapi_app/main.py ./app.py
COPY fastapi_app/config.py ./config.py
COPY data ./data
RUN mkdir -p ./logs
EXPOSE 8000

# Setup an app user so the container doesn't run as the root user
#RUN useradd app
#USER app

# CMD ["python", "fastapi_app/main.py"]
# Это вывешивание портов внутри
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]




# Сбилдить образ
# docker build . -t rsl_fastapi:1.0

# Это вывешивание портов наружу при запуске, должно быть в compose
# docker run -d -p 8000:8000 rsl_fastapi:1.0


# docker run -d -p HOST_PORT:CONTAINER_PORT image
# docker run -P rsl_fastapi:1.0

FROM python:3.11
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
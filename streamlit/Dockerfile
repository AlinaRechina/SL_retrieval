FROM python:3.11
EXPOSE 8501
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD streamlit run app.py
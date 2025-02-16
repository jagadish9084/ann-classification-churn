FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD [ "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
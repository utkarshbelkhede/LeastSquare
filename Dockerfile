FROM python:3.10.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD streamlit run app.py --server.port $PORT
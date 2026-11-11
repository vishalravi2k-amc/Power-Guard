FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["python","-m","streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]

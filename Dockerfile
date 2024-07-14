FROM --platform=$BUILDPLATFORM python:3.9-slim

WORKDIR /cattle_weight

COPY requirements.txt /cattle_weight/requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend_predict.py"]

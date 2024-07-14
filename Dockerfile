FROM --platform=$BUILDPLATFORM python:3.9-slim

WORKDIR /cattle_weight

COPY requirements.txt /cattle_weight/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]
CMD [ "frontend_predict.py", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
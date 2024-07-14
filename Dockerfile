FROM python:3.9-slim

# Use brew for package management on macOS
RUN echo "installing with brew..." && \
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" && \
    brew update && \
    brew install openssl readline sqlite3 xz # potentially needed dependencies for some Python packages

# Install Python dependencies using pip
WORKDIR /cattle_weight

COPY requirements.txt /cattle_weight/requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your project code
COPY . .

EXPOSE 8501

# Use streamlit run for development (adjust script name if needed)
CMD ["streamlit", "run", "frontend_predict.py"]

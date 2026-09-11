FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies without caching pip wheels in container layer
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code
COPY . .

# Install nanogpt CLI package
RUN pip install --no-cache-dir .

ENTRYPOINT ["nanogpt"]
CMD ["--help"]

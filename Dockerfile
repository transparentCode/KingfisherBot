FROM python:3.10.17-slim

WORKDIR /app

# Install build dependencies and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    wget \
    autoconf \
    automake \
    libtool \
    libffi-dev \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib dependencies with updated config scripts
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    wget -O config.sub https://git.savannah.gnu.org/cgit/config.git/plain/config.sub && \
    wget -O config.guess https://git.savannah.gnu.org/cgit/config.git/plain/config.guess && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

COPY pyproject.toml poetry.lock* ./

# Install Poetry and project dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction

COPY . .

EXPOSE 8080

# ENV FLASK_APP=run.py

ENV FLASK_ENV=development

# CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
CMD ["python", "run.py"]
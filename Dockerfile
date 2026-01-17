FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for Pillow/scikit-image)
# libgl1-mesa-glx is sometimes needed for opencv, not necessarily skimage, but good to have basic build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install .

ENV FLASK_APP=dislo_density.web.app
ENV PORT=5000

EXPOSE 5000

CMD ["dislo_density_web", "--host", "0.0.0.0", "--port", "5000"]

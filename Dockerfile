FROM python:3.11 as cora

WORKDIR /app/train_and_predict

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x ./run.sh

CMD ["/bin/bash", "./run.sh"]

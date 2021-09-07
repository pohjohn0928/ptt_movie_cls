FROM python:3.8
WORKDIR /app
COPY ./app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD CUDA_VISIBLE_DEVICES=1 python3 app.py
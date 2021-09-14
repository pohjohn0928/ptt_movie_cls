FROM python:3.8
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8800
CMD python3 app.py
#CUDA_VISIBLE_DEVICES=1

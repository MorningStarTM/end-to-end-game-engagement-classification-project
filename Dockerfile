FROM python:3.11.9-alpine
COPY . /main_db
WORKDIR /main_db
RUN pip install -r requirements.txxt
CMD python main_db.py
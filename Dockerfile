FROM python:3.9-slim

COPY . /code/


RUN pip3 install --no-cache-dir --user pipenv \
  && cd /code \
  && python3.9 -m pipenv install

  
WORKDIR /code
ENTRYPOINT ["python3", "-m", "pipenv", "run", "python", "deltamsi/app.py"]
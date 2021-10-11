FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["churn.py", "./"]

COPY ["models/model1.bin", "models/dv.bin", "models/"]

EXPOSE 9696

ENTRYPOINT ["python", "churn.py"]
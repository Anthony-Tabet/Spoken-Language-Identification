FROM python:3.10.4

WORKDIR /Capstone_project

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . ./

CMD ["python", "app_SLID.py"]
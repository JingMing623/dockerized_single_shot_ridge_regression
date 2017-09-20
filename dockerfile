from python:3.4

WORKDIR /single-shot-ridge-regression

ADD . /single-shot-ridge-regression

RUN pip install -r requirements.txt

# python is fixed program to run the code
ENTRYPOINT ["python"]

# depends on whether local or remote site calls this containter, the CMD part will be overrided with input
CMD ["local.py"]

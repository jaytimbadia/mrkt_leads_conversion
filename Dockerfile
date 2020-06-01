FROM python:3

MAINTAINER Jay Timbadia

RUN mkdir c:\home\lead_predict

COPY data /home/lead_predict/data

COPY model /home/lead_predict/model

COPY version_shift.bat /home/lead_predict/

COPY requirements.txt /home/lead_predict/

RUN pip install --no-cache-dir -r /home/lead_predict/requirements.txt

RUN set PYTHONPATH=/home/lead_predict

CMD [ "python", "/home/lead_predict/run_scripts.py -w 'Predict'" ]
#!/bin/sh


cd /var/www

#uvicorn main:app --host 0.0.0.0 --port 8000 --workers 3

gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:80 --timeout 120

#/usr/bin/supervisord -c /etc/supervisord.conf

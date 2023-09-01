
FROM ubuntu:22.04

LABEL maintainer="SuperDapp"

# Set the working directory in the container
WORKDIR /var/www

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=UTCv
ENV PYTHONUNBUFFERED=1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y software-properties-common

RUN apt-get update \
    && apt -y upgrade \
    && apt-get install -y curl nginx ca-certificates zip unzip git supervisor python3-pip ssl-cert telnet file\
    && apt-get update \
    && apt-get -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy nginx/php/supervisor configs
#RUN cp docker/supervisor.conf /etc/supervisord.conf
#RUN cp docker/nginx.conf /etc/nginx/sites-enabled/default

# Copy the current directory contents into the container at /app
COPY . /var/www

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
#EXPOSE 80

# Run app.py when the container launches
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]


RUN chmod +x /var/www/docker/run.sh

EXPOSE 80

ENTRYPOINT ["/var/www/docker/run.sh"]

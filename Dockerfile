FROM python:3.4-slim

RUN apt-get -y update && \
    apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*


# Install DLIB
RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.7' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS


# Install Flask
RUN cd ~ && \
    pip3 install flask flask-cors


# Install Face-Recognition Python Library
RUN cd ~ && \
    mkdir -p face_recognition && \
    git clone https://github.com/ageitgey/face_recognition.git face_recognition/ && \
    cd face_recognition/ && \
    pip3 install -r requirements.txt && \
    python3 setup.py install

# Install Apache & PHP
RUN cd ~ && \
    apt-get install -y apache2 \
    libapache2-mod-php \
    php \
    php-pear \
    php-mysql \
    php-curl \
    php-gd
#    php-xcache \

# Copy RobinVision python script
RUN cd ~ && \
    mkdir -p app
COPY RobinVision.py /root/app/RobinVision.py
COPY StartRV.sh /root/app/StartRV.sh
COPY config.cfg /root/app/config.cfg

# Copy RobinVision initial encodings
RUN cd ~ && \
    mkdir -p encodings
COPY encodings /root/encodings

# Copy KCFileManager
COPY www /var/www/html

# Give correct authorisation to faces files dir
RUN cd ~ && \ 
     chmod 777 -R /var/www/html/faces && \
     rm -r /var/www/html/index.html

# Start the web service
CMD cd /root/app/ && \
    bash StartRV.sh

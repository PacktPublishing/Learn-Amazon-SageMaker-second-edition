FROM r-base:latest

WORKDIR /opt/ml/
RUN apt-get update
RUN apt-get install -y libcurl4-openssl-dev libsodium-dev

RUN R -e "install.packages(c('rjson', 'plumber'))"

COPY main.R /opt/ml/
COPY train_function.R /opt/ml/
COPY serve_function.R /opt/ml/

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/main.R", "--no-save"]
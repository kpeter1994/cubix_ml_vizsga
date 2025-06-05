FROM ubuntu:latest
LABEL authors="KomPhone"

ENTRYPOINT ["top", "-b"]
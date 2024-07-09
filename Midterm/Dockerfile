FROM python:3.11
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
USER root
RUN apt update
# Dependency for opencv-python (cv2). `import cv2` raises ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# Solution from https://askubuntu.com/a/1015744
RUN apt install -y libgl1-mesa-glx
USER user
RUN pip install -r requirements.txt
COPY . .
CMD ["chainlit", "run", "app.py", "--port", "7860"]
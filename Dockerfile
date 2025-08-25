# Use the official TensorFlow GPU image
FROM tensorflow/tensorflow:2.17.0-gpu

# Set the working directory inside the container
WORKDIR /workspace

# (Optional) Install additional dependencies here
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

# The actual mounting of the current directory is done at runtime with the -v option:
# Example:
# docker build -t skinlesion-gpu .
# docker run --gpus all -it -v %cd%:/workspace skinlesion-gpu

# Container will start in /workspace
CMD ["/bin/bash"]
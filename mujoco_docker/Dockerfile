FROM python:3.8-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    patchelf \
    git \
    curl \
    unzip \
    vim \
    ffmpeg \
    libgl1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libosmesa6 \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    build-essential \
    cmake \
    libopenmpi-dev \
    python3-dev

# Set up environment variables for MuJoCo
# Set environment variables
ENV MUJOCO_VERSION=2.1.1
ENV MUJOCO_DIR=/root/.mujoco/mujoco-2.1.1
ENV MUJOCO_PY_MUJOCO_PATH=${MUJOCO_DIR}
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco-2.1.1/bin:/usr/lib/x86_64-linux-gnu:/usr/lib
ENV MUJOCO_GL=disable

# Set up MuJoCo directory (v2.1.1)
RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget https://github.com/google-deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz && \
    tar -xzf mujoco-2.1.1-linux-x86_64.tar.gz && \
    rm mujoco-2.1.1-linux-x86_64.tar.gz

# Symlink expected library names for mujoco-py
RUN ln -s /root/.mujoco/mujoco2.1.1/lib/libmujoco.so /usr/lib/libmujoco210.so && \
    ln -s /root/.mujoco/mujoco2.1.1/lib/libglewosmesa.so /usr/lib/libglewosmesa.so

RUN ln -s /root/.mujoco/mujoco-2.1.1/lib/libmujoco.so /root/.mujoco/mujoco-2.1.1/bin/libmujoco210.so && \
    ln -s /root/.mujoco/mujoco-2.1.1/lib/libglewosmesa.so /root/.mujoco/mujoco-2.1.1/bin/libglewosmesa.so


# Copy your MuJoCo key
COPY mjkey.txt /root/.mujoco/mjkey.txt
ENV MUJOCO_PY_MJKEY_PATH=/root/.mujoco/mjkey.txt

# Disable GUI rendering (for Rosetta/OpenGL stability)
ENV MUJOCO_GL=osmesa

# Set working directory
WORKDIR /workspace

# Copy requirements first to use Docker cache
COPY requirements.txt .

# Install Python packages from requirements.txt
RUN pip install setuptools==57.5.0 
RUN pip install wheel==0.37.0
RUN pip install pip==23.2
RUN pip install -r requirements.txt

# Copy rest of project files
COPY . .

CMD ["bash"]

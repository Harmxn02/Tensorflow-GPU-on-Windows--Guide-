# Use Tensorflow and Polars with GPU on Windows

Tensorflow does not use your GPU, when running on Windows, because that functionality relies on [cuDF](https://docs.rapids.ai/api/cudf/stable/), which is only available on Linux. Tensorflow used to, but [support for 'native-Windows' was dropped in TensorFlow v2.10](https://www.tensorflow.org/install/pip#windows-native:~:text=Caution%3A%20TensorFlow%202.10%20was%20the%20last%20TensorFlow%20release%20that%20supported%20GPU%20on%20native%2DWindows.%20Starting%20with%20TensorFlow%202.11%2C%20you%20will%20need%20to%20install%20TensorFlow%20in%20WSL2%2C%20or%20install%20tensorflow%20or%20tensorflow%2Dcpu%20and%2C%20optionally%2C%20try%20the%20TensorFlow%2DDirectML%2DPlugin). The latest version is 2.17, and does not support it :(

In this README you will get a step-by-step guide on how to create a Docker Linux container where Tensorflow **does** have access to training with a GPU. In some cases, during my testing, training with my GPU _(NVIDIA RTX 3060 Laptop, 6GB vram)_ was 30 times faster than training on just my CPU.

If you plan to use `polars[gpu]` (instead of pandas), it is around 10 times faster than Polars without GPU. And Polars in general is already 10x faster than Pandas. [Pandas also has a way of using GPU](https://rapids.ai/cudf-pandas/), and you will need to follow this guide for that too (mentioned all the way at the bottom)

## Guide

### Step 1: Prerequisites

| Software             | Installation link                                                                                              |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| Docker Desktop       | <https://www.docker.com/products/docker-desktop/>                                                                |
| NVIDIA CUDA Toolkit  | <https://developer.nvidia.com/cuda-downloads?target_os=Windows>                                                  |
| NVIDIA cuDNN Library | <https://developer.nvidia.com/cudnn-downloads>                                                                   |
| WSL                  | no link, just install a Linux distro from Windows Store, I did it with Debian, but Ubuntu should work the same |
|                      |                                                                                                                |

### Step 2: Setting up prerequisites

#### NVIDIA software

1. Launch and install it
2. Restart system after you installed both of them

#### WSL (normally you should already have done this in other courses)

1. browse to `Control Panel` > `Programs` > `Turn Windows features on or off` > turn on checkbox that says 'Windows Subsystem for Linux'
2. Launch the downloaded distro, and make a new user
3. Close out of the Linux terminal and open up a Powershell or Command Prompt terminal, type this:

```bash
wsl --set-default-version 2 # this you might not have done, definitely do this!
```

#### Docker Desktop

1. Launch it and go to `Settings` > `Resources` > check if WSL 2 is being used
2. Close it

It should say something like "You are using the WSL 2 backend, so resource limits are managed by Windows."

### Step 3: Dockerfile

The Dockerfile is included in this folder

Create a textfile (.txt) by right-clicking anywhere in a folder of your choice, and replace the full name to `Dockerfile`. Windows should ask you if you are sure you want to change the file type. If your filename is `Dockerfile.txt` you did not change the file type, and you should remove the `.txt` from the file name

Here are the contents, but you should get the file included, so just use that:

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /workspace

# Install necessary dependencies including Git
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install jupyter jupyterlab numpy pandas matplotlib seaborn scikit-learn && \
    pip install tensorflow[and-cuda] polars[gpu]

# Expose Jupyter port
EXPOSE 8888

# Set the command to keep the container running for VS Code
CMD ["tail", "-f", "/dev/null"]
```

### Step 4: Build and Run the Dockerfile

> As an example I will call my container **`dev-container-env`**, but you can call it something else

#### A. Build

Open a terminal in the folder where you stored the Dockerfile, and run this command:

```bash
docker build -t dev-container-env .
```

If you stored it on your desktop the Terminal path should be `PS C:\Users\yourname\Desktop\`, but you should store it in a safe location, because you might need it again

don't forget the period after the container name. There is also a space between the container name and the period

#### B. Run

After building, which will take forever, because Tensorflow's GPU version is extremely large, you can run it. We will use the `--gpus all` flag, which is **extremely important** not to forget. If you forget it, you will not use your GPU's, which is against the whole point of this guide

```bash
docker run -it --gpus all -p 8888:8888 -v ${PWD}:/workspace dev-container-env
```

If you ever have to re-run this command, it will tell you the port 8888:8888 is already in use. Simply go into Docker Desktop and stop the container which is running on that port, then re-run the command.

After running this command, your terminal should create a new blank line under the command, and it will look like your terminal froze. Open Docker Desktop and check if you have a running container on the port 8888:8888. If yes, you can close the terminal.

### Step 5: VS Code 'dev containers'

1. Open VS Code
2. Go to the extensions tab and install:
    - Dev Containers (by Microsoft)
    - Remote Development (by Microsoft)
    - Docker (by Microsoft)

3. Right-click on the sidebar-menu-thingie and enable `Remote Explorer`, so that it always shows up in your sidebar
4. Click on `Remote Explorer`.
5. In the `DEV CONTAINERS` tab, expand `Dev Containers` to find your Docker container.
    - If you hover over the name of a Docker container, you will see an arrow pointing right, and a folder with a plus-sign. Click either of them, your choice.

> There may be multiple containers with the same name. Next to each container there is a "codename" which is set in Docker. Open Docker Desktop and check the codename for the Docker container which is running on port 8888:8888

### Step 6: Done

- You will now be inside of a dev container, and you can now run the `.ipynb` notebooks to test if everything worked
- Normally there should only be 1 Python version installed, so pick that one.
- You can create a virtual environment, it advises you to do that whenever you `pip install` something, but I can't be bothered.
- If something ever goes wrong with the dev container, you can re-run all the commands above. But remember, you will lose all files inside the dev container.
- The Dockerfile comes with GIT, and it works using your device's configurations by default (Thank the developers at Microsoft for that)

- **Pandas**: run this command in the top cell of your notebook. [Guide for this](https://rapids.ai/cudf-pandas/)

``` python
%load_ext cudf.pandas
import pandas as pd
```

> **If you have ever made projects that trained on Tensorflow, `git clone` them and see if they train faster now (they should)**

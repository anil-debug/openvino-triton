# OpenVINO Model Deployment with Docker

This guide outlines the steps to deploy an OpenVINO model using Docker with NVIDIA Triton Inference Server.

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker installed on your system
- OpenVINO model(s) available in a local directory

## Steps

1. **Run the Docker command to deploy the model**:

    ```bash
    docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/kumar/openvino_models:/models nvcr.io/nvidia/tritonserver:24.03-py3 tritonserver --model-repository=/models
    ```

    - `--gpus=1`: Specifies the number of GPUs to be used (adjust as needed).
    - `-p8000:8000 -p8001:8001 -p8002:8002`: Maps the Triton Inference Server's REST API endpoints to host ports.
    - `-v /path/to/openvino_models:/models`: Mounts the local directory containing OpenVINO models to the container.

2. **Access the deployed model**:

    Once the Docker container is running, you can access the deployed model through Triton Inference Server's REST API. 

    Example API endpoints:
    - Model metadata: `http://localhost:8000/v2/models`
    - Model inference: `http://localhost:8000/v2/models/<model_name>/infer`

    Replace `<model_name>` with the actual name of your model.

## Additional Notes

- Ensure that your Docker daemon has access to the NVIDIA runtime by following the NVIDIA Docker installation instructions.
- Make sure your system meets the hardware and software requirements for running OpenVINO and NVIDIA Triton Inference Server.

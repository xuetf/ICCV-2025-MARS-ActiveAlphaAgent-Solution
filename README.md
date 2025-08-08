# ICCV 2025 Submission

This repository contains the code for our submission to the **MARS2 Workshop @ ICCV 2025**.

## 🏆 Competition Results

We are proud to announce our achievements in the [Multimodal Reasoning Competition](https://lens4mllms.github.io/mars2-workshop-iccv2025/):

*   🥇 **1st Place** in Track 1: **VG-RS** (Visual Grounding in Real-world Scenarios)
*   🥉 **3rd Place** in Track 2: **VQA-SA** (Visual Question Answering with Spatial Awareness)
*   🥉 **3rd Place** in Track 3: **VR-Ads** (Visual Reasoning in Creative Advertisement Videos)

This repository provides the code to reproduce our results for all three tracks.

## Model Zoo

We have released the models for the **VG-RS** and **VQA-SA** tracks on Hugging Face. You can download them from the following links:

*   **VG-RS Model**: [Zach996/ActiveAlphaAgent-VG-RS](https://huggingface.co/Zach996/ActiveAlphaAgent-VG-RS)
*   **VQA-SA Model**: [Zach996/ActiveAlphaAgent-VQA-SA](https://huggingface.co/Zach996/ActiveAlphaAgent-VQA-SA)

To download the models, you can use `git lfs`:
```bash
# Make sure you have git-lfs installed
# git lfs install

# Clone the repository for the VG-RS model
git clone https://huggingface.co/Zach996/ActiveAlphaAgent-VG-RS

# Clone the repository for the VQA-SA model
git clone https://huggingface.co/Zach996/ActiveAlphaAgent-VQA-SA
```

## Setup Environment

### Hardware Requirements
*   **GPU**: The experiments are conducted on servers equipped with 8x NVIDIA A100 (80G) or H800 (80G) GPUs. The provided scripts are configured for an 8-GPU setup.

### Prerequisites
*   **Python Version**: This project requires Python `3.10` or higher.
*   **PyTorch**: Ensure you have a compatible version of PyTorch installed for your CUDA environment.
*   **Dependencies**: Install all required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### Key Dependencies
This project relies on several key libraries. Below are some of the most important ones, with their versions specified in `requirements.txt`:
*   `torch==2.6.0+cu124`
*   `transformers==4.51.3`
*   `vllm==0.8.2`
*   `flash-attn==2.7.4.post1`
*   `xformers==0.0.29.post2`
*   `decord==0.6.0` & `pyav==14.2.1` (for video decoding)

## 1. Visual Grounding on VG-RS

This task performs visual grounding on the VG-RS dataset.

### Inference via Script

We provide a convenient script to run the entire inference pipeline.

1.  **Configure the script**: Open `run_grounding.sh` and modify the variables in the "配置" (Configuration) section to match your environment, especially `INFERENCE_MODE`, `MODEL_PATH`, and data paths.

2.  **Run the script**:
    ```bash
    bash run_grounding.sh
    ```
The script will handle both `hf` and `client` modes. In `client` mode, it will automatically manage the VLLM service (start it if not running, check health, and use an existing service if available).

## 2. Visual Question Answering on VQA-SA

This task performs VQA on the VQA-SA dataset.

### Step 1: Data Preprocessing (Optional)

For context-aware VQA (`--prompt_version v2`), the `run_vqa.sh` script will automatically check for and generate a question file with context if it doesn't exist. You just need to ensure the original question file (e.g., `VQA-SA-question.json`) is present at the path specified in the script.

### Step 2: Inference via Script

1.  **Configure the script**: Open `run_vqa.sh` and modify the variables in the "配置" (Configuration) section to match your environment, especially `INFERENCE_MODE`, `MODEL_PATH`, and data paths.

2.  **Run the script**:
    ```bash
    bash run_vqa.sh
    ```
The script handles both `hf` and `client` modes, automatically managing the VLLM service in `client` mode.

## 3. Video Question Answering on VR-Ads

This task performs VQA on the VR-Ads dataset.

### Inference via Script

1.  **Configure the script**: Open `run_video_reasoning.sh` and modify the variables in the "配置" (Configuration) section to match your environment, especially `MODEL_PATH` and data paths.

2.  **Run the script**:
    ```bash
    bash run_video_reasoning.sh
    ```
The script will first check if a compatible VLLM service is already running. If not, it will start one, wait for it to be ready, and then proceed with the inference. After the task is complete, it will remind you to stop the service if it was started by the script.

## Argument Descriptions

### Common Arguments (`eval_grounding_vqa.py`)

*   `--json_path`: Path to the input JSON file containing evaluation data.
*   `--task`: Specifies the task to run, choices are `grounding` or `vqa`.
*   `--output_dir`: Directory to save the output results.
*   `--image_base_dir`: The root directory where images are stored.
*   `--model_name`: A name for your model configuration, used to generate the default output filename.
*   `--inference_mode`: The inference framework, choices are `hf`, `vllm`, `client`.
*   `--model_path`: Path to the trained model checkpoint. (Applicable for `hf` and `vllm` modes).
*   `--gpu_ids`: Comma-separated list of GPU IDs to use for inference. (Applicable for `hf` mode).
*   `--port`: The port number for the API service. (Applicable for `client` mode).
*   `--num_workers`: Number of worker threads for data processing in `client` mode.
*   `--prompt_version`: The prompt version for the VQA task, choices are `v1` or `v2`. (Applicable for `vqa` task).
*   `--min_pixels`, `--max_pixels`: The minimum/maximum number of pixels for image resizing during preprocessing.
*   `--output_path`: Specifies the full path for the output JSON file. If not provided, it will be automatically generated in `--output_dir` based on the input filename and model name.

### Video VQA Arguments (`eval_video_reasoning.py`)

*   `--api_url`: The API endpoint for the VLLM server.
*   `--model_name`: The name of the model being evaluated.
*   `--video_root_path`: The root directory where video files are stored.
*   `--question_file_path`: The full path to the JSON file containing questions.
*   `--output_dir`: Directory to save the output results.
*   `--fps`: Frames per second to sample from the video.
*   `--max_workers`: Number of worker threads for data processing.
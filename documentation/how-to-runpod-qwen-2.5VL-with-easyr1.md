# Running EasyR1 on a RunPod Instance: A Step-by-Step Guide

This guide walks you through setting up and running the EasyR1 code on your RunPod instance without using Docker. You will:
1. Install Git and clone the repository into `/data`
2. Set up a Python virtual environment and install dependencies reliably
3. Run an example training script with an example dataset
4. Monitor NVIDIA GPU memory usage while the script loads

---

## 1. Pre-requisites

Ensure that you have the following:

- **Sudo privileges** on your RunPod instance.
- **Python 3.9+** installed.
- **NVIDIA GPUs** with proper drivers installed.
- **Git** (if not already installed, we will install it).
- (Optional) The `watch` command to monitor GPU memory (usually available on most systems, or install via your package manager).

---

## 2. Installing Git & Cloning the Repository

Open a terminal on your RunPod instance and execute the following commands:

Update package index and install git (if not already installed)
```bash 
apt update && apt install git -y
```

Change directory to /data (where you want to clone the repository)
```bash
cd /data
```

Clone the EasyR1 repository from GitHub
```bash
git clone https://github.com/hiyouga/EasyR1.git
```


This will create a new directory `/data/EasyR1` containing the codebase.

---

## 3. Setting Up the Python Environment

It is recommended to use a virtual environment to avoid dependency conflicts.

Navigate to the cloned repository and set up a virtual environment:


Change directory to the repository
```bash
cd /data
```

Create a virtual environment (using Python's built-in venv)
```bash
python3 -m venv easyr1-venv
```

Activate the virtual environment
```bash
source easyr1-venv/bin/activate
```

Upgrade pip to the latest version
```bash
pip install --upgrade pip
```

# Install wheel

```bash
pip install wheel
```

---

## 4. Installing the Project Dependencies

The most reliable method is to use the provided `requirements.txt` file and install the package in editable mode.

Change directory to the repository
```bash
cd /data/EasyR1
```

Install the packaging module
```bash
pip install packaging
```

# Install latest torch
```bash
pip install torch torchvision torchaudio
```

Install all required packages (including accelerate, codetiming, datasets, vllm, transformers, etc.)
```bash
# Install requirements.txt but EXCLUDE flash-attn which we'll install separately
pip install -r <(grep -v "flash-attn" requirements.txt)
```

# Install flash-attn correctly to avoid CUDA compatibility issues
# We use a specific pre-compiled wheel that's compatible with PyTorch 2.5.x and CUDA 12
```bash
pip uninstall -y flash-attn transformer-engine
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Install the EasyR1 package in editable mode
```bash
pip install -e .
```

---

## 5. Install the Missing `mathruler` Dependency

EasyR1 requires the `mathruler` package (used in `verl/utils/reward_score/math.py`), which is not included automatically in the `requirements.txt`. To install it, run:

```bash
pip install git+https://github.com/hiyouga/mathruler.git
```

This ensures that all dependencies are correctly installed and that any changes to the code are immediately reflected.

---

## 6. Move Cache Directories to Data Volume

To preserve disk space on the container and utilize the larger data volume for model storage, move cache directories:

```bash
# Create cache directory on data volume
mkdir -p /data/cache-models/
mkdir -p /data/cache-models/huggingface
mkdir -p /data/cache-models/modelscope
mkdir -p /data/cache-models/huggingface/hub
mkdir -p /data/cache-models/modelscope/hub

# Move modelscope cache
rm -rf mv /root/.cache/modelscope /data/cache-models/
ln -s /data/cache-models/modelscope /root/.cache/modelscope

# Move huggingface cache
rm -rf /root/.cache/huggingface /data/cache-models/
ln -s /data/cache-models/huggingface /root/.cache/huggingface

# Verify symlinks
ls -la /root/.cache/
```

This moves the model caches to your larger data volume and creates symlinks from the original locations.

---

## 7. Running an Example Script

EasyR1 comes with example shell scripts located in the `examples/` folder. To run a test example using the provided math dataset, execute:

First, edit the script to reflect the correct number of GPUs you have available.

```bash
sed -i 's/trainer.n_gpus_per_node=4/trainer.n_gpus_per_node=8/' examples/run_qwen2_5_7b_math.sh
```


Then run the script:

```bash
bash examples/run_qwen2_5_7b_math.sh
```


> **Note:**
>
> - The script sets the environment variable `VLLM_ATTENTION_BACKEND` to `XFORMERS` for efficient attention computation.
> - The `MODEL_PATH` variable in the script should point to your local model path. Adjust it as needed.
> - The example uses the [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k) dataset. Make sure your instance can reach external URLs to download the dataset if needed.
> - If you disable wandb with `trainer.use_wandb=false`, the training metrics will only be displayed in the console output.

If you would like to try the version that uses the SwanLab logger, run:

```bash
bash examples/run_qwen2_5_7b_math_swanlab.sh
```

---

## 8. Monitoring NVIDIA GPU Memory

To monitor the NVIDIA GPU memory usage while the script loads and runs, open a new terminal session (or use a multiplexer like tmux/screen) and run:

```bash
watch -n 1 nvidia-smi
# or
watch -n 1 "nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,memory.total,memory.used,memory.free --format=csv,noheader,nounits"
```

# or
```bash
watch -n 1 "echo 'GPU   Total(MiB)   Used(MiB)'; nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{printf \"%-3s %-12s %-10s\n\", \$1, \$2, \$3}'"
```



This command updates the GPU status every second, allowing you to keep an eye on memory utilization in real time.

---

## 9. Summary

To recap:

1. **Clone the Repository:**  
   Install Git (if needed) and clone EasyR1 into `/data`.

2. **Setup & Install Dependencies:**  
   Create and activate a Python virtual environment, then install dependencies using a modified approach for flash-attention compatibility.

3. **Run an Example Script:**  
   Use one of the provided example shell scripts (e.g., `run_qwen2_7b_math.sh`) to start a test training run using an example dataset.

4. **Monitor NVIDIA GPU Memory:**  
   Use `watch -n 1 nvidia-smi` in a separate terminal to watch the GPU memory usage as the script loads.

You're now ready to explore and experiment with EasyR1 on your RunPod instance. For any further details or customizations, refer to the repository's README and documentation.

Happy training!




# Setting Up RunPod with 1000GB Storage and SWIFT Framework and Your Own SSH Key

This guide walks through setting up a RunPod instance with 1000GB storage and installing the SWIFT framework.

## Prerequisites

- RunPod account with sufficient credits
- SSH client installed on your local machine
- Basic familiarity with Linux commands

## 1. Create 500GB-1000GB Storage Volume

1. Log into your RunPod account
2. Navigate to **Volumes** section
3. Click **Create Volume**
4. Configure the volume:
   - Name: `data` (or your preferred name)
   - Size: `500` (GB)
   - Select your preferred datacenter
5. Click **Create**

## 2. Create and Configure Pod

1. Navigate to **Pods** section
2. Click **+ Deploy**
3. Select **GPU** type based on your needs
4. Set your GPU count to more than 1 if you need multiple GPUs
5. Selection your template: the default Pytorch template with Ubuntu-22.04 is enough
6. Click "Edit Template" 
   - Configure pod settings:
   - Choose Ubuntu-based template
   - Set container disk size
   - Under **Volume**, attach your created 500GB volume:
     - Rename your Mount path: `/data`
   - Enable public IP
   - Set your container disk size to 50GB

## 3. Ensure "ssh" and "start Jupyter Notebook" are checked

## 4. Generate Local SSH Key

Run these commands on your local machine to generate your SSH key:

```bash
# Generate SSH key
ssh-keygen -t my_runpod_key -C "your_email@example.com"

# Display public key to copy
cat ~/.ssh/my_runpod_key.pub
```

## 4. Add SSH Key to Pod

1. Under https://www.runpod.io/console/user/settings
2. Find **SSH Public Keys** section
3. Paste your copied public key

## 5. Deploy Instance

1. Review all configurations
2. Click **Deploy**
3. Wait for pod to initialize
4. Note the assigned IP address and SSH port

## 6. Connect to your Pod via SSH

Connect to your pod via SSH

1. Click "Connect" on your runpod deployed instance.
2. Copy the "SSH over exposed TCP: (Supports SCP & SFTP)" field:

It should something like this:

ssh root@216.81.245.23 -p 12046 -i ~/.ssh/id_ed25519

Now change it to connect using the generated key:

```bash
# Connect using the generated SSH key
ssh -i ~/.ssh/my_runpod_key root@<POD_IP_ADDRESS> -p <SSH_PORT>
```

It should looke like this, if we use the example IP and port from above:
```bash
ssh -i ~/.ssh/my_runpod_key root@216.81.245.23 -p 12046
```

## 7. Set Up Python Virtual Environment
Create and activate virtual environment:

```bash
# Update system packages
apt update && apt upgrade -y

# Install Python tools
apt install -y python3-pip python3-venv python3-dev vim curl wget htop zip unzip tar gzip build-essential libopenmpi-dev pkg-config git jq 


# Create virtual environment
```bash
cd /data
python3 -m venv swift-env

# Activate environment
source swift-env/bin/activate
```

If you are running on a NON NVLINK environment, such as NVIDIA A40s, you will need to export these variables:

NCCL_P2P_DISABLE=1 
NCCL_DEBUG=WARN 
NCCL_IB_DISABLE=1 


```bash
# Upgrade pip
python3 -m ensurepip --default-pip
pip install --upgrade pip

# Install wheel

```bash
pip install wheel
```

## If you are training:

# Install SWIFT with all capabilities. You will need the nightly build. 

# nightly build
```bash
pip install --upgrade "git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]" --upgrade-strategy only-if-needed

For reference, the standard install is:
# standard install
```bash
pip install 'ms-swift[all]' -U
```

# Verify installation
```bash
python3 -c "import swift; print(swift.__version__)"
```


# Install DeepSpeed
```bash
pip install deepspeed
```

# You will need the math verify library with Ms-Swift if you are using reasoning datasets with GRPO:
```bash
pip install math_verify
```

# If you need debugging to see what is going on
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

# Install Qwen Utils if you need Qwen2 or 2.5 VL models

```bash     
pip install qwen-vl-utils

# Nightly build of vLLM

```bash
pip uninstall -y vllm; pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

# For reference the standard install is:

```bash
pip install vllm
```

# LIKELY YOU DO NOT NEED THIS AS IT IS ALREADY installed by the above vllm command but it is here for reference:

# Install latest torch
# Install baseline offical release with no CUDA indicated (will already likely be installed by the above vllm command)
```bash
pip install torch torchvision torchaudio
```

# Install latest transformers

For Qwen 2.5 VL models you may need this until official release:
```bash
pip install --upgrade --force-reinstall git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
```


# Then install flash-attn from PyPI

```bash
pip install flash-attn --no-build-isolation
```


## 9. Move Cache Directories to Data Volume

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

### Optional Dependencies

# Nightly build of transformers

## Checking GPU Status

1. **Checking GPU Status**
   ```bash
   watch -n 1 nvidia-smi
   # or
   watch -n 1 "nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,memory.total,memory.used,memory.free --format=csv,noheader,nounits"
   ```

   # or
   ```bash
   watch -n 1 "echo 'GPU   Total(MiB)   Used(MiB)'; nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{printf \"%-3s %-12s %-10s\n\", \$1, \$2, \$3}'"
   ```

2. **Monitoring Storage**
   ```bash
   df -h /data
   ```

## References

- [RunPod Documentation](https://docs.runpod.io)([1](https://docs.runpod.io))
- [SWIFT Framework Documentation](https://swift.readthedocs.io)([2](https://swift.readthedocs.io))
- [ModelScope Documentation](https://modelscope.cn)([3](https://modelscope.cn))
- [Training Llama 3.1 with Swift](https://www.shelpuk.com/post/fine-tuning-llama-3-1-with-swift-unsloth-alternative-for-multi-gpu-llm-training)

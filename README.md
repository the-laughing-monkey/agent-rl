# agent-rl Training Scripts

Welcome to the **agent-rl** repository! This repo contains training scripts for Reinforcement Learning with GRPO (Group Relative Policy Optimization) using the ms‑swift framework (v3) along with bleeding edge versions of Hugging Face Transformers and vLLM.

## Requirements

- **Python 3.8+**
- **ms‑swift (v3):** Install the nightly build to access all capabilities.
- **Hugging Face Transformers:** Nightly build for Qwen 2.5 VL model support.
- **vLLM:** Nightly build for Qwen 2.5 VL model support
- **DeepSpeed:** For zero3 sharding.
- **Additional Dependencies:**  
  - `math_verify` (for verifying math reasoning datasets with GRPO)  
    ```bash
    pip install math_verify
    ```
  - `qwen-vl-utils` *(for Qwen 2.5 VL models)*
    ```bash
    pip install qwen-vl-utils
    ```

## Setup and Installation

For detailed instructions on setting up a RunPod instance with 1000GB storage, the SWIFT framework, and support for Qwen 2.5 VL models, please refer to:
[How to RunPod with Qwen 2.5 VL Models](documentation/how-to-runpod-qwen-2.5VL-models.md)

## Usage

Refer to the training scripts in the `scripts_train/` directory for various configurations:
- **Full Training with vLLM**
- **LoRA Training with/without vLLM**

Each script sets critical training parameters such as batch sizes, number of generations, and reward functions. Check the comments within each script for further details.

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [ms‑swift GRPO Documentation](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO.md)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [vLLM Documentation](https://vllm.ai)
- [Swift Framework Documentation](https://swift.readthedocs.io)
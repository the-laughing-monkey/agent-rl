# agent-rl Training Scripts

Welcome to the **agent-rl** repository! This repo contains training scripts for Reinforcement Learning with GRPO (Group Relative Policy Optimization) using the ms‑swift framework (v3) along with bleeding edge versions of Hugging Face Transformers and vLLM.


<div align="center">
  <img src="https://github.com/the-laughing-monkey/agent-rl/blob/main/images/robo-workout-1-smaller.jpg?raw=true" alt="Robo Workout">
</div>
## Requirements

See below guides for different RL Frameworks (ms-swift or EasyR1) and model sizes (Qwen 2.5 VL or Qwen 2.5)

## Setup and Installation

For detailed instructions on setting up a RunPod instance with 1000GB storage, the SWIFT framework, and support for Qwen 2.5 VL models, please refer to:
[How to RunPod with Qwen 2.5 VL Models](documentation/how-to-runpod-qwen-2.5VL-models.md)

For instructions on setting up and running EasyR1 (a reinforcement learning framework for LLMs) on a RunPod instance with Qwen 2.5 models, please refer to:
[How to Run EasyR1 with Qwen 2.5 Models on RunPod](documentation/how-to-runpod-qwen-2.5VL-with-easyr1.md)

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
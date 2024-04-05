# A Minimal Example of Fine-Tuning Language Models

This repository contains a minimal example of fine-tuning autoregressive language models (e.g. Llama 2) on a language modeling task without LoRA. By default, it uses Hugging Face's [`accelerate`](https://huggingface.co/docs/accelerate/en/index) and [DeepSpeed ZeRO-3](https://huggingface.co/docs/transformers/main/en/deepspeed) to fine-tune the model on two GPUs.

The code has only been tested on a machine with two NVIDIA A100 GPUs. If you have a different setup, you may need to adjust the code accordingly.

## Configuration

Make sure to use the latest version of `transformers` and install `accelerate` amd `deepspeed` before proceeding.

```shell
python -m pip install -U transformers
python -m pip install accelerate deepspeed
```

The `accelerate_config.yaml` configuration file can also be obtained by running `accelerate config` from the command line. This will start an interactive prompt that will guide you through the process of creating a configuration file.

For the DeepSpeed configuration, please refer to the [DeepSpeed documentation](https://www.deepspeed.ai/docs/config-json/) for a comprehensive list of options.

## Usage

To fine-tune the model, run the following command:

```shell
accelerate launch \
  --config_file accelerate_config.yaml \
  --use_deepspeed \
  --deepspeed_config_file deepspeed_config.json \
  finetune.py \
  --dataset_name wikitext \
  --subset_name wikitext-2-raw-v1 \
  --model_name meta-llama/Llama-2-7b-hf \
  --num_epochs 5 \
  --lr 2e-5 \
  --batch_size 4
```

## Notes

- Do not use `device_map` or `model.to(device)` to move the model to a specific device. Accelerate will handle this for you.
- I am using the `paged_adamw_32bit` optimizer here, because the regular `adamw_hf` won't fit on two GPUs. Note that the paged version could be slower. Feel free to try if [other optimizers](https://github.com/huggingface/transformers/blob/76fa17c1663a0efeca7208c20579833365584889/src/transformers/training_args.py#L146) work for you.

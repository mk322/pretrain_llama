# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import argparse

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from model import ModelArgs, Transformer
from generate import LLaMA

from transformers import AutoTokenizer

os.environ['OMP_NUM_THREADS'] = '2'


def load(
    param_path: str,
    # local_rank: int,
    # world_size: int,
    ckpt_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    
    # save hard-coded param
    # params = {}
    # params['dim'] = 2048
    # params['n_heads'] = 4
    # params['n_layers'] = 4
    # params['vocab_size'] = 32000
    # params['multiple_of'] = 256
    # params['norm_eps'] = 1e-5

    # with open(param_path, 'w') as f:
    #     json.dump(params, f)

    start_time = time.time()
    # checkpoints = sorted(Path(param_path).glob("*.pth"))
    # assert world_size == len(
    #     checkpoints
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    # ckpt_path = checkpoints[local_rank]

    print("Loading")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # load params
    with open(param_path, "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = "<unk>"
    model_args.vocab_size = tokenizer.vocab_size
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    param_path: str = "/gscratch/zlab/haoqik/pretrain_llama/out/training/params.json",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 64,
):

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    ckpt_path = "true_out/training/iter-000006-ckpt.pth"
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-l','--load_iter_path', type=str, default="out/training/iter-000004-ckpt.pth")
    # args = parser.parse_args()
    # load_file_path = args.load_iter_path
    # if not os.path.isfile(load_file_path):
    #     raise Exception(f'{load_file_path} is not a valid file path')
    # else:
    #      ckpt_path = load_file_path

    generator = load(
        param_path, ckpt_path, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
        Sentiment: Negative
        ###
        Tweet: "My day has been 👍"
        Sentiment: Positive
        ###
        Tweet: "This is the link to the article"
        Sentiment: Neutral
        ###
        Tweet: "This new music video was incredibile"
        Sentiment:""",
        """Translate English to French:

        sea otter => loutre de mer

        peppermint => menthe poivrée

        plush girafe => girafe peluche

        cheese =>""",
    ]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")

def setup_model_parallel():
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))

        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)
        return local_rank, world_size

if __name__ == "__main__":
    fire.Fire(main)
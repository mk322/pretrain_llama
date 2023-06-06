import sys
import os
import torch
import fire
import time
import json
import lightning as L
torch.set_float32_matmul_precision('high')
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from model import ModelArgs, Transformer
from generate import LLaMA
from transformers import AutoTokenizer
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO
param_path = "params.json"
ckpt_path = "iter-000004-ckpt.pth"
temperature: float = 1
top_p: float = 1
def load(
    #fabric, 
    param_path: str,
    ckpt_path: str,
) -> LLaMA:

    start_time = time.time()

    print("Loading")

    checkpoint = torch.load(ckpt_path, map_location="cuda:0")

    # load params
    with open(param_path, "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        **params
    )    
    new_state_dict = OrderedDict()
    model = Transformer(model_args)
    for k, v in checkpoint.items():
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = "<unk>"
    tokenizer.pad_token_id = 0
    model = Transformer(model_args)

    model_args.vocab_size = 32000
    torch.set_default_tensor_type(torch.cuda.HalfTensor)


    torch.set_default_tensor_type(torch.FloatTensor)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main():
    local_rank, world_size = setup_model_parallel()

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")


    generator = load(param_path, ckpt_path)

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "",
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Jeanette Sawyer Cohen, PhD, ",
        "I've learned",
        "But something is happening here and you don’t know what it is Do you, Mr, Jones?"
    ]
    results = generator.generate(
        prompts, max_gen_len=20, temperature=temperature, top_p=top_p
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

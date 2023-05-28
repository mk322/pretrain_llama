import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from model import Transformer, ModelArgs, TransformerBlock  # Assuming model.py is in the same directory
import os
import time
import tqdm
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from preprocess_data import load_data
#from lit_llama.utils import save_model_checkpoint

def setup_model_parallel():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

local_rank, world_size = setup_model_parallel()

auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=TransformerBlock)

fabric = L.Fabric(accelerator="cuda", devices=2, precision="bf16-mixed", strategy=strategy)
fabric.launch()
fabric.seed_everything(1337 + fabric.global_rank)

#with fabric.device:
model_args = fabric.to_device(ModelArgs(dim=4096, n_layers=2, n_heads=2, vocab_size=320))  # Update these parameters as needed
model = fabric.to_device(Transformer(model_args))

model = fabric.setup_module(model)

epochs = 1
log_interval = 1
max_iters = 10
out_dir = "out/training"
eval_interval = 20

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Load the dataset
train_data, valid_data, test_data = load_data()

#def preprocess_function(examples):
    #return tokenizer(examples['text'], truncation=True, max_length=512)

#encoded_dataset = dataset.map(preprocess_function, batched=True)

#dataloader = DataLoader(encoded_dataset, batch_size=8, shuffle=True)
optimizer = torch.optim.Adam(model.parameters())
optimizer = fabric.setup_optimizers(optimizer)
#device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else torch.device('cpu')
#model.to(device)
#print(device)

# If multiple GPUs are available, use them
#if torch.cuda.device_count() > 1:
    #print("Let's use", torch.cuda.device_count(), "GPUs!")
    #model = nn.DataParallel(model)
    #model = model.to(device)

# Initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def validate(model, validation_dataset):
    validation_loss = 0
    with torch.no_grad():
        for batch in validation_dataset:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            validation_loss += outputs.loss.item()
    return validation_loss / len(validation_dataset)

print("start_init")
model.apply(initialize_weights)
print("end_init")



def train(
    fabric,
    model,
    optimizer: torch.optim.Optimizer,
    train_data,
    val_data):
    iter_num = 0

    while True:
        # TODO: add learning rate scheduling

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = validate(fabric, model, val_data)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            fabric.print(f"Saving checkpoint to {out_dir}")
            #save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))

        t0 = time.time()
        for i, batch in enumerate(train_data):
            input_ids = torch.stack(batch['input_ids'], dim=0)
            fabric.to_device((input_ids.pin_memory()))

        logits = model(input_ids)
        loss = logits[0]
        #loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        fabric.backward(loss)

        # TODO: Gradient clipping
        # if grad_clip != 0.0:
        #     fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
        iter_num += 1

        if iter_num > max_iters:
            break

train(fabric, model, optimizer, train_data, valid_data)


'''
# Training loop
for epoch in range(epochs):
    print("in_epoch")
    for i, batch in enumerate(train_data):
        print("start!!!")
        if i > 100:
            continue
        optimizer.zero_grad()
        #batch = {k: v.to(device) for k, v in batch.items()}
        print(len(batch['input_ids']))
        input_ids = torch.stack(batch['input_ids'], dim=0).to(device)

        #attention_mask = torch.Tensor(batch['attention_mask']).to(device)
        outputs = model(input_ids).to(device)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        print("yeah!!!")
    print("finish batch!!!")
    validation_loss = validate(model, valid_data)
    print(f'Validation loss: {validation_loss}')
'''





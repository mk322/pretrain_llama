# pretrain_llama
inference:
torchrun --nproc_per_node=1 --master-port=29501 inference.py --num_nodes=1

pretrain:
torchrun --nproc_per_node=1 --master-port=29501 train.py --num_nodes=1

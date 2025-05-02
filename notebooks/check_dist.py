import torch
import torch.distributed as dist
import torchmetrics.metric
import os
# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 /home/lmh/projects_dir/MulConf0420/notebooks/check_dist.py
class myMetric(torchmetrics.metric.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[])#, dist_reduce_fx=None
        # self.add_state("preds", default=[], dist_reduce_fx="cat")
    def update(self, preds):
        self.preds.append(preds)
    def compute(self):
        print("hello")
        return self.preds
def main():
    # Initialize the process group (this will vary depending on the backend, in this case, NCCL)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # Create a tensor to gather from each rank
    rank = dist.get_rank()
    metric = myMetric()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # tensor = torch.tensor([rank, rank+1], device=device)
    tensor = torch.ones(rank, device=device)
    # tensor = torch.tensor([rank, rank+1], device=device)
    metric.update(tensor)
    tensor = torch.zeros(rank, device=device)
    metric.update(tensor)
    print(f"Rank {rank} tensor: {tensor}")
    print(f"Rank {rank} metric: {metric.compute()}")
    # Clean up the distributed process group
    dist.destroy_process_group()    
# def all_gather_example(tensor):
#     # Get the world size and rank
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
    
#     # Create a list to store the tensors from all processes
#     tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
#     # Use all_gather to collect tensors from all processes
#     dist.all_gather(tensor_list, tensor)
    
#     print(f"Rank {rank} gathered tensors: {tensor_list}")

# def main():
#     # Initialize the process group (this will vary depending on the backend, in this case, NCCL)
#     dist.init_process_group(backend='nccl', init_method='env://')
    
#     # Create a tensor to gather from each rank
#     rank = dist.get_rank()
    
#     device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
#     tensor = torch.tensor([rank, rank+1], device=device)

#     # Call the all_gather_example function
#     all_gather_example(tensor)
    
#     # Clean up the distributed process group
#     dist.destroy_process_group()



if __name__ == "__main__":
    main()

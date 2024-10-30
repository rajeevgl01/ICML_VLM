import os
import builtins
import torch
import math
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def setup_for_distributed(is_master, world_size):
    """
    This function disables printing when not in the master process (rank 0)
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.now().time()
            builtin_print(f'[{now}]', *args, **kwargs)

    builtins.print = print

# Setup DDP environment
def setup_ddp(rank, world_size):
    dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=dist_url)
    setup_for_distributed(rank == 0, world_size)

# Cleanup DDP
def cleanup_ddp():
    dist.destroy_process_group()

# Save checkpoints with flexible inputs
def save_checkpoint(image_model, text_model, optimizer, scaler, epoch, best_test_loss, avg_test_loss, test_auc, checkpoint_dir, name):
    checkpoint = {
        'epoch': epoch + 1,
        'imaget_model_state_dict': image_model.state_dict(),
        'text_model_state_dict': text_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_test_loss': best_test_loss,
        'test_loss': avg_test_loss,
        'test_auc': test_auc
    }
    filename = os.path.join(checkpoint_dir, f"{name}.pt")
    torch.save(checkpoint, filename)

# Load checkpoint and resume
def load_checkpoint(checkpoint_dir, image_model, text_model, optimizer, scaler):
    checkpoint = torch.load(checkpoint_dir, map_location='cpu')
    image_model.load_state_dict(checkpoint['image_model_state_dict'])
    text_model.load_state_dict(checkpoint['text_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    return start_epoch, checkpoint['best_test_loss'], checkpoint['test_auc']

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if hasattr(args, 'fixed_lr') and args.fixed_lr:
            lr = args.lr
        else:
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
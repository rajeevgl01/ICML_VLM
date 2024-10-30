import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import timm.optim.optim_factory as optim_factory
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os
import wandb
from datetime import datetime
import argparse
from utils import *
from engine import *
from adapter import AdapterClassifier

# custom dataloader
def load_data(pt_file):
    data = torch.load(pt_file, map_location='cpu')
    embeddings = data['embeddings']  # shape: [num_samples, embedding_dim]
    labels = data['labels']          # shape: [num_samples, num_classes]
    return embeddings, labels

# Training function
def main(rank, world_size, args):
    # Initialize wandb for logging
    if rank == 0:
        wandb.init(project="ICML_text_backbone", config=args)

    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Set device for the current process
    device = torch.device(f'cuda:{rank}')

    # Create dataloaders
    train_embeddings, train_labels = load_data(args.train_file)
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=False)
    
    test_embeddings, test_labels = load_data(args.test_file)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4, pin_memory=True, drop_last=False)
    
    # Initialize model, optimizer, loss function, and NativeScaler for mixed precision
    model = AdapterClassifier(args.input_dim, args.output_dim, args.num_classes).to(device)
    print(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    model_without_ddp = model
    model = DDP(model, device_ids=[rank])
    criterion = nn.BCEWithLogitsLoss()
    
    eff_batch_size = args.batch_size * world_size
    args.lr = args.lr * eff_batch_size / 256

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = optim.AdamW(param_groups, lr=args.lr)
    scaler = NativeScalerWithGradNormCount()  

    best_loss = float('inf')
    best_auc = 0.0

    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.exists(checkpoint_path):
            start_epoch, best_loss, best_auc = load_checkpoint(checkpoint_path, model, optimizer, scaler)
            print(f"Resumed from checkpoint at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        avg_train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, scaler, device, rank, epoch, args.num_epochs, args)

        print(f"Rank {rank} Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {avg_train_loss}")

        # Test after every epoch
        avg_test_loss, test_auc = test(model, test_dataloader, criterion, device, rank, args)
        print(f"Rank {rank} Epoch [{epoch+1}/{args.num_epochs}], Test Loss: {avg_test_loss}, AUC: {test_auc}")

        # Log metrics to wandb (only rank 0)
        if rank == 0:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'test_auc': test_auc,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Checkpoint for the latest model
        if rank == 0:
            save_checkpoint(model_without_ddp, optimizer, scaler, epoch, best_loss, avg_test_loss, test_auc, args.checkpoint_dir, "latest")

            # Save checkpoint for the best loss
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                save_checkpoint(model_without_ddp, optimizer, scaler, epoch, best_loss, avg_test_loss, test_auc, args.checkpoint_dir, "best_loss")

            # Save checkpoint for the best AUC
            if test_auc > best_auc:
                best_auc = test_auc
                save_checkpoint(model_without_ddp, optimizer, scaler, epoch, best_loss, avg_test_loss, best_auc, args.checkpoint_dir, "best_auc")
    
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Training with Checkpointing and wandb Logging")
    parser.add_argument('--train_file', type=str, required=True, help="Path to the train .pt file containing embeddings and labels")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the test .pt file containing embeddings and labels")
    parser.add_argument('--checkpoint_root_dir', type=str, default='./checkpoints', help="Root directory to save checkpoints (with date/time appended)")
    parser.add_argument('--resume', default=False, action='store_true', help="Resume training from latest checkpoint if available")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of warmup epochs to train the model")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate for training")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate for the CosineAnnealingLR scheduler")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for Adam optimizer")
    parser.add_argument('--num_classes', type=int, default=5, help="number of classes in the dataset")
    parser.add_argument('--input_dim', type=int, default=4096, help="Embedding dimensions of the text embedder model")
    parser.add_argument('--output_dim', type=int, default=768, help="Feature dimension; must be same as the image feature extraction backbon. Eg: 768 for ViT-B/16")

    args = parser.parse_args()

    # Run the main function for multi-GPU training
    world_size = torch.cuda.device_count()  # Number of available GPUs

    if int(os.environ['RANK']) == 0:
        # Add date and time to checkpoint directory
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.checkpoint_dir = os.path.join(args.checkpoint_root_dir, f"run_{current_time}")

        os.makedirs(args.checkpoint_dir, exist_ok=True)

    main(int(os.environ['RANK']), world_size, args)


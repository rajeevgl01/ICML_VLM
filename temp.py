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
from engine_pretrain_text import *
from torch.utils.data import Dataset
from adapter import AdapterClassifier

class MedicalDataset(Dataset):
	def __init__(self, pt_file):
		self.data = torch.load(pt_file, map_location='cpu')

		self.file_paths = []

		for key, val in self.data.items():
			self.file_paths.append(key)
	
	def __len__(self):
		return len(self.file_paths)
		
	def __getitem__(self, idx):
		item = self.data[self.file_paths[idx]]
		return self.file_paths[idx], item['embedding']

def train_one_epoch(text_model, dataloader, device, rank, epoch, num_epochs):
	text_model.eval()

	if rank == 0:
		progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", position=0)
	else:
		progress_bar = enumerate(dataloader)
	
	logit_dict = {}

	# Inference loop
	for _, (file_path, embedding) in progress_bar:
		embedding = embedding.to(device, non_blocking=True)

		# Mixed precision training with gradient scaler
		with torch.cuda.amp.autocast(dtype=torch.bfloat16):
			text_logits, _ = text_model(embedding)
			logit_dict[file_path[0]] = nn.functional.sigmoid(text_logits)[0].detach().cpu()
			print(nn.functional.sigmoid(text_logits)[0])
	
	torch.save(logit_dict, "/home/local/ASURITE/rgoel15/ICML_VLM/passage_logits.pt")

# Training function
def main(rank, world_size, args):
	# Setup DDP
	setup_ddp(rank, world_size)
	
	# Set device for the current process
	device = torch.device(f'cuda:{rank}')

	# Create dataloaders
	train_dataset = MedicalDataset(args.train_file)
	train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=False)
	
	# Initialize model, optimizer, loss function, and NativeScaler for mixed precision
	model = AdapterClassifier(args.input_dim, args.output_dim, args.num_classes).to(device)
	print(str(model))
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print('number of params (M): %.2f' % (n_parameters / 1.e6))
	model_without_ddp = model
	model = DDP(model, device_ids=[rank])

	checkpoint = torch.load(args.resume)
	model_without_ddp.load_state_dict(checkpoint['model_state_dict'], strict=True)
	
	train_dataloader.sampler.set_epoch(0)
	train_one_epoch(model, train_dataloader, device, rank, 0, args.num_epochs)


	cleanup_ddp()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Multi-GPU Training with Checkpointing and wandb Logging")
	parser.add_argument('--train_file', type=str, required=True, help="Path to the train .pt file containing embeddings and labels")
	parser.add_argument('--resume', default=None, help="Resume training from latest checkpoint if available")
	parser.add_argument('--eval', default=None, help="Checkpoint path to evaluate the model")
	parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs to train the model")
	parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of warmup epochs to train the model")
	parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training")
	parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate for training")
	parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate for the CosineAnnealingLR scheduler")
	parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for Adam optimizer")
	parser.add_argument('--num_classes', type=int, default=5, help="number of classes in the dataset")
	parser.add_argument('--input_dim', type=int, default=4096, help="Embedding dimensions of the text embedder model")
	parser.add_argument('--output_dim', type=int, default=768, help="Feature dimension; must be same as the image feature extraction backbon. Eg: 768 for ViT-B/16")

	args = parser.parse_args()

	# Run the main function for multi-GPU training
	world_size = torch.cuda.device_count()  # Number of available GPUs

	main(int(os.environ['RANK']), world_size, args)


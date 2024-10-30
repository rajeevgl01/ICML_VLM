import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import timm.optim.optim_factory as optim_factory
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataloader import build_dataset_chest_xray
import os
import wandb
from datetime import datetime
import argparse
from utils import *
from engine import *
from adapter import AdapterFeatures
import models_vit
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import lr_decay as lrd

# Training function
def main(rank, world_size, args):
	# Initialize wandb for logging
	if rank == 0:
		wandb.init(project="ICML_finetune_IB", config=args)

	# Setup DDP
	setup_ddp(rank, world_size)
	
	# Set device for the current process
	device = torch.device(f'cuda:{rank}')

	# Define file paths for centroids and anti-centroids
	paths = {
		'image_centers': 'image_centers_chexpert.pth',
		'image_anti_centers': 'image_anti_centers_chexpert.pth',
		'embedding_centers': 'embedding_centers_chexpert.pth',
		'embedding_anti_centers': 'embedding_anti_centers_chexpert.pth'
	}

	# Check if all necessary files exist
	if all(os.path.exists(path) for path in paths.values()):
		# Load centroids and anti-centroids from files
		image_centers = torch.load(paths['image_centers'], map_location='cpu')
		image_anti_centers = torch.load(paths['image_anti_centers'], map_location='cpu')
		embedding_centers = torch.load(paths['embedding_centers'], map_location='cpu')
		embedding_anti_centers = torch.load(paths['embedding_anti_centers'], map_location='cpu')
	else:
		# Calculate centroids and anti-centroids
		image_centers, image_anti_centers, embedding_centers, embedding_anti_centers = get_input_centroids(train_dataloader, device, rank, args)

		# Save the computed centroids and anti-centroids to files
		torch.save(image_centers, paths['image_centers'])
		torch.save(image_anti_centers, paths['image_anti_centers'])
		torch.save(embedding_centers, paths['embedding_centers'])
		torch.save(embedding_anti_centers, paths['embedding_anti_centers'])

	# Create dataloaders
	train_dataset = build_dataset_chest_xray('train', image_centers, image_anti_centers, embedding_centers, embedding_anti_centers, args)
	test_dataset = build_dataset_chest_xray('test', image_centers, image_anti_centers, embedding_centers, embedding_anti_centers, args)
	
	train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=False)

	test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4, pin_memory=True, drop_last=False)
	
	# Initialize model, optimizer, loss function, and NativeScaler for mixed precision
	text_backbone = AdapterFeatures(args.input_dim, args.output_dim).to(device)
	n_parameters = sum(p.numel() for p in text_backbone.parameters() if p.requires_grad)
	print('number of params in text_backbone (M): %.2f' % (n_parameters / 1.e6))

	image_backbone = models_vit.__dict__[args.image_model](
			img_size=args.input_size,
			num_classes=args.num_classes,
			drop_rate=args.vit_dropout_rate,
			drop_path_rate=args.drop_path,
			global_pool=args.global_pool,
		)
	
	if args.finetune and not args.eval:
		checkpoint = torch.load(args.finetune, map_location='cpu')

		print(f"Load pre-trained checkpoint from: {args.finetune}")
		checkpoint_model = checkpoint['model']
		state_dict = image_backbone.state_dict()
		for k in ['head.weight', 'head.bias']:
			if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
				print(f"Removing key {k} from pretrained checkpoint")
				del checkpoint_model[k]
		if args.global_pool:
			for k in ['fc_norm.weight', 'fc_norm.bias']:
				try:
					del checkpoint_model[k]
				except:
					pass

		interpolate_pos_embed(image_backbone, checkpoint_model)
		msg = image_backbone.load_state_dict(checkpoint_model, strict=False)
		print(msg)
		trunc_normal_(image_backbone.head.weight, std=2e-5)
	
	if args.finetune_adapter and not args.eval:
		checkpoint = torch.load(args.finetune_adapter, map_location='cpu')

		print(f"Load pre-trained adapter checkpoint from: {args.finetune_adapter}" )
		checkpoint_model = checkpoint['model_state_dict']
		temp = checkpoint_model.copy()
		for k in temp.keys():
			if 'head' in k:
				del checkpoint_model[k]
		
		msg = text_backbone.load_state_dict(checkpoint_model, strict=True)
		print(msg)
	
	image_backbone.to(device)
	text_backbone.to(device)

	image_backbone_without_ddp = image_backbone
	text_backbone_without_ddp = text_backbone
	image_backbone = DDP(image_backbone, device_ids=[rank])
	text_backbone = DDP(text_backbone, device_ids=[rank])
	criterion = nn.BCEWithLogitsLoss()
	
	eff_batch_size = args.batch_size * world_size
	args.lr = args.lr * eff_batch_size / 256
	args.text_lr = args.text_lr * eff_batch_size / 256

	text_backbone_param_groups = optim_factory.param_groups_weight_decay(text_backbone_without_ddp, args.text_backbone_weight_decay)
	image_backbone_param_groups = lrd.param_groups_lrd(image_backbone_without_ddp, args.weight_decay,
			no_weight_decay_list=image_backbone_without_ddp.no_weight_decay(),
			layer_decay=args.layer_decay
		)

	for param_group in text_backbone_param_groups:
		param_group['lr'] = args.text_lr

	for param_group in image_backbone_param_groups:
		param_group['lr'] = args.lr

	# Combine the parameter groups and pass them to the optimizer
	optimizer = optim.AdamW(image_backbone_param_groups + text_backbone_param_groups)
	scaler = NativeScalerWithGradNormCount()  

	best_loss = float('inf')
	best_auc = 0.0
	early_stopping = 0

	start_epoch = 0
	if args.resume:
		checkpoint_path = os.path.join(args.checkpoint_dir, "latest.pt")
		if os.path.exists(checkpoint_path):
			start_epoch, best_loss, best_auc = load_checkpoint(checkpoint_path, image_backbone, text_backbone, optimizer, scaler)
			print(f"Resumed from checkpoint at epoch {start_epoch}")

	# Training loop
	for epoch in range(start_epoch, args.num_epochs):
		if early_stopping == 20:
			exit()

		train_dataloader.sampler.set_epoch(epoch)
		image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids = compute_feature_centroids(image_backbone, text_backbone, train_dataloader, device, rank, args)
		avg_train_loss, ce_loss, image_IB_loss, text_IB_loss = train_one_epoch(image_backbone, text_backbone, train_dataloader, 
																		 criterion, optimizer, scaler, device, rank, epoch, args.num_epochs, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, args)

		print(f"Rank {rank} Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {avg_train_loss}")

		# Test after every epoch
		avg_test_loss, test_auc = test(image_backbone, test_dataloader, criterion, device, rank, args)
		print(f"Rank {rank} Epoch [{epoch+1}/{args.num_epochs}], Test Loss: {avg_test_loss}, AUC: {test_auc}")

		max_lr = 0.
		for group in optimizer.param_groups:
			max_lr = max(max_lr, group["lr"])
		# Log metrics to wandb (only rank 0)
		if rank == 0:
			wandb.log({
				'epoch': epoch + 1,
				'train_loss': avg_train_loss,
				'test_loss': avg_test_loss,
				'ce_loss': ce_loss,
				'image_ib_loss': image_IB_loss,
				'text_ib_loss': text_IB_loss,
				'test_auc': test_auc,
				'lr': max_lr
			})

		if test_auc < best_auc:
			early_stopping += 1
		else:
			early_stopping = 0

		# Checkpoint for the latest model
		if rank == 0:
			save_checkpoint(image_backbone_without_ddp, text_backbone_without_ddp, optimizer, scaler, epoch, best_loss, avg_test_loss, test_auc, args.checkpoint_dir, "latest")

			# Save checkpoint for the best loss
			if avg_test_loss < best_loss:
				best_loss = avg_test_loss
				save_checkpoint(image_backbone_without_ddp, text_backbone_without_ddp, optimizer, scaler, epoch, best_loss, avg_test_loss, test_auc, args.checkpoint_dir, "best_loss")

			# Save checkpoint for the best AUC
			if test_auc > best_auc:
				best_auc = test_auc
				save_checkpoint(image_backbone_without_ddp, text_backbone_without_ddp, optimizer, scaler, epoch, best_loss, avg_test_loss, best_auc, args.checkpoint_dir, "best_auc")
	
	cleanup_ddp()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Multi-GPU Training with Checkpointing and wandb Logging")
	# dataloader args
	parser.add_argument('--train_file', type=str, required=True, help="Path to the train .pt file containing embeddings and labels")
	parser.add_argument('--train_embedding_file', type=str, required=True, help="Path to the train .pt file containing embeddings and labels")
	parser.add_argument('--test_file', type=str, required=True, help="Path to the test .pt file containing embeddings and labels")
	parser.add_argument('--checkpoint_root_dir', type=str, default='./checkpoints', help="Root directory to save checkpoints (with date/time appended)")
	parser.add_argument('--data_path', default=None, required=True, type=str, help='dataset root path')
	parser.add_argument('--resume', default=False, action='store_true', help="Resume training from latest checkpoint if available")
	# training args
	parser.add_argument('--num_epochs', type=int, default=75, help="Number of epochs to train the model")
	parser.add_argument('--input_size', type=int, default=224, help="input image size")
	parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of warmup epochs to train the model")
	parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training")
	parser.add_argument('--lr', type=float, default=2.5e-4, help="Initial learning rate for training")
	parser.add_argument('--text_lr', type=float, default=2.5e-5, help="Initial learning rate for training")
	parser.add_argument('--min_lr', type=float, default=1e-5, help="Minimum learning rate for the CosineAnnealingLR scheduler")
	parser.add_argument('--text_backbone_weight_decay', type=float, default=0.05, help="text backbone weight decay for Adam optimizer")
	parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay for Adam optimizer")
	parser.add_argument('--num_classes', type=int, default=5, help="number of classes in the dataset")
	parser.add_argument('--eval', type=bool, default=False, help="")
	# text backbone args
	parser.add_argument('--input_dim', type=int, default=4096, help="Embedding dimensions of the text embedder model")
	parser.add_argument('--output_dim', type=int, default=768, help="Feature dimension; must be same as the image feature extraction backbon. Eg: 768 for ViT-B/16")
	parser.add_argument('--finetune_adapter', required=True, type=str, help='path to pretrained adapter model')
	# image backbone args
	parser.add_argument('--image_model', default='vit_base_patch16', type=str, help='Name of image backbone model to train')
	parser.add_argument('--finetune', required=True, type=str, )
	parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate for vit (default: 0.1)')
	parser.add_argument('--global_pool', action='store_true')
	parser.add_argument('--layer_decay', type=float, default=0.55, help='Layer-wise lr decay from ELECTRA/BEiT')
	parser.set_defaults(global_pool=True)
	parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')
	parser.add_argument('--vit_dropout_rate', type=float, default=0, help='Dropout rate for ViT blocks (default: 0.0)')
	# Information bottleneck weight
	parser.add_argument('--image_ib_weight', default=10, type=float)
	parser.add_argument('--text_ib_weight', default=10, type=float)

	args = parser.parse_args()

	# Run the main function for multi-GPU training
	world_size = torch.cuda.device_count()  # Number of available GPUs

	if int(os.environ['RANK']) == 0:
		# Add date and time to checkpoint directory
		current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		args.checkpoint_dir = os.path.join(args.checkpoint_root_dir, f"run_{current_time}")

		os.makedirs(args.checkpoint_dir, exist_ok=True)

	main(int(os.environ['RANK']), world_size, args)


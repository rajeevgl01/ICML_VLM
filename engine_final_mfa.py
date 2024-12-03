import torch
import torch.nn as nn
from utils import adjust_learning_rate
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch.distributed as dist
import scipy

def computeAUROC(dataGT, dataPRED, classCount):
	outAUROC = []
	# print(dataGT.shape, dataPRED.shape)
	for i in range(classCount):
		try:
			outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
		except:
			outAUROC.append(0.)
	print(outAUROC)
	return outAUROC

@torch.no_grad()
def test(model, dataloader, criterion, device, rank, args):
	model.eval()
	targets = []
	preds = []
	test_loss = []
	if rank == 0:
		progress_bar = tqdm(dataloader, total=len(dataloader), desc="Testing", position=0)
	else:
		progress_bar = dataloader

	for input, target in progress_bar:
		input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
		with torch.cuda.amp.autocast(dtype=torch.bfloat16):
			outputs = model(input)[0]
			loss = criterion(outputs, target)
		outputs = outputs.to(torch.float32)
		test_loss.append(loss.item())

		if rank == 0:
			progress_bar.set_postfix(loss=loss.item())

		preds.append(outputs)
		targets.append(target)

	preds = torch.cat(preds, dim=0).sigmoid().cpu().numpy()
	targets = torch.cat(targets, dim=0).cpu().numpy()

	auc_each_class = np.array(computeAUROC(targets, preds, args.num_classes))
	auc_avg = np.mean(auc_each_class[auc_each_class != 0])

	avg_test_loss = sum(test_loss) / len(test_loss)
	return avg_test_loss, auc_avg

def trucated_nuclear_loss(F, U, V, softRank, id_F):
	if U.shape[0] != F.shape[0]:
		U_mini_batch = U[id_F]
	S = U_mini_batch.T @ F
	S = S @ V
	S = torch.diag(S)
	nuc_loss = torch.norm(S[softRank:], p=1)
	return nuc_loss

def train_one_epoch(image_model, text_model, text_subnet, dataloader, criterion, optimizer, scaler, device, rank, epoch, num_epochs, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, U, V, args):
	assert U.numel() > 0
	assert V.numel() > 0

	image_model.train()
	text_model.train()
	text_subnet.eval()
	epoch_loss = []
	bc_loss = []
	image_IB = []
	text_IB = []
	low_rank_loss = []
	optimizer.zero_grad()

	if rank == 0:
		progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", position=0)
	else:
		progress_bar = enumerate(dataloader)

	# Inference loop
	for batch_idx, (input, embedding, targets, input_score, embedding_score, _, indexes) in progress_bar:
		adjust_learning_rate(optimizer, batch_idx / len(dataloader) + epoch, args)

		# Transfer data to GPU
		input = input.to(device, non_blocking=True)
		embedding = embedding.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		input_score = input_score.to(device, non_blocking=True)
		embedding_score = embedding_score.to(device, non_blocking=True)
		indexes = indexes.to(device, non_blocking=True)

		# Mixed precision training with gradient scaler
		with torch.cuda.amp.autocast(dtype=torch.bfloat16):
			image_logits, image_features = image_model(input)
			text_logits, text_features = text_model(embedding)
			with torch.no_grad():
				text_features_weights = text_subnet(text_features)
				text_features_weights = torch.where(text_logits > args.threshold, 1, text_features_weights)
				text_features_weights = text_features_weights.clone().detach()
				weighted_text_features = text_features_weights.unsqueeze(1).permute(0, 2, 1) * text_features.unsqueeze(2).permute(0, 2, 1) / text_features_weights.sum(dim=0).unsqueeze(1)
			# weighted_text_features = text_features
			# Compute cross-entropy loss for this batch and accumulate
			ce_loss = criterion(image_logits, targets)

		image_ib_loss, text_ib_loss = get_IB(0, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 0].squeeze(1), input_score, embedding_score, 0, 0, device)
		image_ib_loss, text_ib_loss = get_IB(1, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 1].squeeze(1), input_score, embedding_score, image_ib_loss, text_ib_loss, device)
		image_ib_loss, text_ib_loss = get_IB(2, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 2].squeeze(1), input_score, embedding_score, image_ib_loss, text_ib_loss, device)
		image_ib_loss, text_ib_loss = get_IB(3, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 3].squeeze(1), input_score, embedding_score, image_ib_loss, text_ib_loss, device)
		image_ib_loss, text_ib_loss = get_IB(4, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 4].squeeze(1), input_score, embedding_score, image_ib_loss, text_ib_loss, device)
		
		tn_loss = trucated_nuclear_loss(image_features, U, V, args.soft_rank, indexes.to(torch.int64))
		loss = ce_loss + args.tn_loss_weight * tn_loss + args.image_ib_weight * image_ib_loss / 5 + args.text_ib_weight * text_ib_loss / 5

		# Scale and backpropagate the final loss
		scaler(loss, optimizer, clip_grad=None, parameters=concat_generators(image_model.parameters(), text_model.parameters()), create_graph=False, update_grad=True)

		epoch_loss.append(loss.item())
		bc_loss.append(ce_loss.item())
		image_IB.append(image_ib_loss.item() / 5)
		text_IB.append(text_ib_loss.item() / 5)
		low_rank_loss.append(tn_loss.item())

		optimizer.zero_grad()
		torch.cuda.synchronize()

		if rank == 0:
			progress_bar.set_postfix(loss=loss.item(), ce_loss=ce_loss.item(), image_IB_loss=image_ib_loss.item() / 5, text_IB_loss=text_ib_loss.item() / 5, tn_loss=tn_loss.item())

	k = len(epoch_loss)
	return sum(epoch_loss) / k, sum(bc_loss) / k, sum(image_IB) / k, sum(text_IB) / k, sum(low_rank_loss) / k

def compute_feature_centroids(image_model, text_model, subnet_model, subnet_optimizer, scaler, dataloader, device, rank, epoch, args):
	# Put models in evaluation mode
	image_model.eval()
	text_model.eval()
	subnet_model.train()
	subnet_optimizer.zero_grad()
	adjust_learning_rate(subnet_optimizer, epoch, args)

	if rank == 0:
		progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Compute Feature Centroids", position=0)
	else:
		progress_bar = enumerate(dataloader)

	# Initialize accumulators for features on CPU
	local_image_features = torch.empty((0, )).to(device)
	local_text_features = torch.empty((0, )).to(device)
	local_labels = torch.empty((0, )).to(device)
	local_indexes = torch.empty((0, )).to(device)
	local_input_scores = torch.empty((0, )).to(device)
	local_embedding_scores = torch.empty((0, )).to(device)
	local_logits = torch.empty((0, )).to(device)

	for batch_idx, (input, embedding, targets, input_score, embedding_score, text_logits, indexes) in progress_bar:
		# Transfer data to GPU
		input = input.to(device, non_blocking=True)
		embedding = embedding.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		indexes = indexes.to(device, non_blocking=True)
		input_score = input_score.to(device, non_blocking=True)
		embedding_score = embedding_score.to(device, non_blocking=True)
		text_logits = text_logits.to(device, non_blocking=True)

		with torch.cuda.amp.autocast(dtype=torch.bfloat16) and torch.no_grad():
			# Forward pass for feature extraction
			image_feature = image_model(input)[1]  # Assuming image_model returns logits and features
			_, text_feature = text_model(embedding)   # Assuming text_model only returns features

		# Append results to CPU tensors
		local_image_features = torch.cat((local_image_features, image_feature), dim=0)
		local_text_features = torch.cat((local_text_features, text_feature), dim=0)
		local_labels = torch.cat((local_labels, targets), dim=0)
		local_indexes = torch.cat((local_indexes, indexes), dim=0)
		local_input_scores = torch.cat((local_input_scores, input_score), dim=0)
		local_embedding_scores = torch.cat((local_embedding_scores, embedding_score), dim=0)
		local_logits = torch.cat((local_logits, text_logits), dim=0)

	# Prepare to gather features and scores from all GPUs (on CPU)
	gather_image_features = [torch.zeros_like(local_image_features) for _ in range(dist.get_world_size())]
	gather_text_features = [torch.zeros_like(local_text_features) for _ in range(dist.get_world_size())]
	gather_labels = [torch.zeros_like(local_labels) for _ in range(dist.get_world_size())]
	gather_indexes = [torch.zeros_like(local_indexes) for _ in range(dist.get_world_size())]
	gather_input_scores = [torch.zeros_like(local_input_scores) for _ in range(dist.get_world_size())]
	gather_embedding_scores = [torch.zeros_like(local_embedding_scores) for _ in range(dist.get_world_size())]
	gather_logits = [torch.zeros_like(local_logits) for _ in range(dist.get_world_size())]

	# Gather all the features, labels, and scores across all GPUs
	dist.all_gather(gather_image_features, local_image_features)
	dist.all_gather(gather_text_features, local_text_features)
	dist.all_gather(gather_labels, local_labels)
	dist.all_gather(gather_indexes, local_indexes)
	dist.all_gather(gather_input_scores, local_input_scores)
	dist.all_gather(gather_embedding_scores, local_embedding_scores)
	dist.all_gather(gather_logits, local_logits)

	# Concatenate the gathered results from all processes (on CPU)
	image_features = torch.cat(gather_image_features, dim=0)
	text_features = torch.cat(gather_text_features, dim=0)
	labels = torch.cat(gather_labels, dim=0)
	indexes = torch.cat(gather_indexes)
	input_score = torch.cat(gather_input_scores)
	embedding_score = torch.cat(gather_embedding_scores)
	logits = torch.cat(gather_logits).detach()

	with torch.cuda.amp.autocast(dtype=torch.bfloat16):
		subnet_weights = subnet_model(text_features)
	
		mask = logits > args.threshold
		weights = torch.where(mask, 1, subnet_weights)
				
		weighted_text_features = weights.unsqueeze(1).permute(0, 2, 1) * text_features.unsqueeze(2).permute(0, 2, 1) / weights.sum(dim=0).unsqueeze(1)

		# Compute centroids and anti-centroids with data from all GPUs on CPU
		image_feature_centroids, image_feature_anti_centroids = compute_centroids(image_features, labels, args.num_classes)
		text_feature_centroids, text_feature_anti_centroids = compute_centroids(weighted_text_features, labels, args.num_classes)

		image_ib_loss, text_ib_loss = get_IB(0, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 0].squeeze(1), input_score, embedding_score, 0, 0, device)
		image_ib_loss, text_ib_loss = get_IB(1, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 1].squeeze(1), input_score, embedding_score, image_ib_loss, text_ib_loss, device)
		image_ib_loss, text_ib_loss = get_IB(2, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 2].squeeze(1), input_score, embedding_score, image_ib_loss, text_ib_loss, device)
		image_ib_loss, text_ib_loss = get_IB(3, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 3].squeeze(1), input_score, embedding_score, image_ib_loss, text_ib_loss, device)
		_, text_ib_loss = get_IB(4, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, weighted_text_features[:, 4].squeeze(1), input_score, embedding_score, image_ib_loss, text_ib_loss, device)

		mse = F.mse_loss(weights, subnet_weights)
	
	loss = text_ib_loss + mse
	print(f'loss={loss.item()}, mse_loss={mse.item()}, text_ib_loss={text_ib_loss.item()}')
	scaler(loss.float(), subnet_optimizer, clip_grad=None, parameters=subnet_model.parameters(), create_graph=False, update_grad=True)
	# Free memory for feature tensors after computing centroids
	del text_features, labels, gather_image_features, gather_text_features, gather_labels, gather_indexes
	del local_image_features, local_indexes, local_labels, local_text_features
	torch.cuda.empty_cache()

	return image_feature_centroids.detach(), image_feature_anti_centroids.detach(), text_feature_centroids.detach(), text_feature_anti_centroids.detach(), image_features, indexes

def get_IB(cls, image_feature_centroids, image_feature_anti_centroids, text_feature_centroids, text_feature_anti_centroids, image_features, text_features, inputs_score, embeddings_score, image_ib_loss, text_ib_loss, device):
	image_feature_cls_centroid = image_feature_centroids[cls].unsqueeze(0)
	image_feature_cls_anti_centroid = image_feature_anti_centroids[cls].unsqueeze(0)
	text_feature_cls_centroid = text_feature_centroids[cls].unsqueeze(0)
	text_feature_cls_anti_centroid = text_feature_anti_centroids[cls].unsqueeze(0)

	image_cls_features_scores = compute_cluster_scores(image_features, torch.cat([image_feature_cls_centroid, image_feature_cls_anti_centroid], dim=0), device)
	text_cls_features_scores = compute_cluster_scores(text_features, torch.cat([text_feature_cls_centroid, text_feature_cls_anti_centroid], dim=0), device)

	Q_image = torch.matmul(image_cls_features_scores.T, text_cls_features_scores) / torch.sum(text_cls_features_scores, dim=0)
	Q_text = torch.matmul(text_cls_features_scores.T, image_cls_features_scores) / torch.sum(image_cls_features_scores, dim=0)
	
	# Compute the IB loss for this class
	IB_image_cls = get_IB_Loss(inputs_score[:, cls], image_cls_features_scores, text_cls_features_scores, Q_image)
	IB_text_cls = get_IB_Loss(embeddings_score[:, cls], text_cls_features_scores, image_cls_features_scores, Q_text)
	return image_ib_loss + IB_image_cls, text_ib_loss + IB_text_cls

def concat_generators(*kwargs):
	for gen in kwargs:
		yield from gen

@torch.no_grad()
def get_input_centroids(dataloader, device, rank, args):
	if rank == 0:
		progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Cluster Centroids", position=0)
	else:
		progress_bar = enumerate(dataloader)

	# Function to update local centroids and anti-centroids
	def update_local_centroids_and_anti_centroids(centroids, anti_centroids, data, labels, num_classes):
		for d, lbl in zip(data, labels):
			for cls in range(num_classes):
				if lbl[cls] == 1.:
					centroids[cls]['sum'] += d
					centroids[cls]['count'] += 1
				elif lbl[cls] == 0.:
					anti_centroids[cls]['sum'] += d
					anti_centroids[cls]['count'] += 1

		return centroids, anti_centroids

	# Function to aggregate centroids across all GPUs using all-reduce
	def all_reduce_centroids_and_anti_centroids(centroids, anti_centroids, num_classes):
		for cls in range(num_classes):
			# Aggregate the sum and count across all GPUs for centroids
			dist.all_reduce(centroids[cls]['sum'], op=dist.ReduceOp.SUM)
			dist.all_reduce(centroids[cls]['count'], op=dist.ReduceOp.SUM)

			# Aggregate the sum and count across all GPUs for anti-centroids
			dist.all_reduce(anti_centroids[cls]['sum'], op=dist.ReduceOp.SUM)
			dist.all_reduce(anti_centroids[cls]['count'], op=dist.ReduceOp.SUM)

		return centroids, anti_centroids

	# Function to compute the final centroids and anti-centroids after global aggregation
	def compute_final_centroids_and_anti_centroids(centroids, anti_centroids, num_classes, data_size):
		centroids_list = []
		anti_centroids_list = []

		for cls in range(num_classes):
			if centroids[cls]['count'].item() > 0:
				centroid = centroids[cls]['sum'] / centroids[cls]['count']
			else:
				centroid = torch.zeros((data_size), device=centroids[cls]['sum'].device)
			centroids_list.append(centroid)

			if anti_centroids[cls]['count'].item() > 0:
				anti_centroid = anti_centroids[cls]['sum'] / anti_centroids[cls]['count']
			else:
				anti_centroid = torch.zeros((data_size), device=anti_centroids[cls]['sum'].device)
			anti_centroids_list.append(anti_centroid)

		# Stack the list into tensors
		centroids_tensor = torch.stack(centroids_list)
		anti_centroids_tensor = torch.stack(anti_centroids_list)

		return centroids_tensor, anti_centroids_tensor

	# Define the size of image data and embeddings
	image_size = 224 * 224 * 3  # Assuming flattened images of size 224x224x3
	embedding_size = 4096  # Embedding size as given

	# Initialize centroids and anti-centroids for both image data and embeddings
	centroids_images = {
		cls: {
			'sum': torch.zeros((image_size), device=device),
			'count': torch.tensor(0.0, device=device)
		}
		for cls in range(args.num_classes)
	}
	anti_centroids_images = {
		cls: {
			'sum': torch.zeros((image_size), device=device),
			'count': torch.tensor(0.0, device=device)
		}
		for cls in range(args.num_classes)
	}

	centroids_embeddings = {
		cls: {
			'sum': torch.zeros((embedding_size), device=device),
			'count': torch.tensor(0.0, device=device)
		}
		for cls in range(args.num_classes)
	}
	anti_centroids_embeddings = {
		cls: {
			'sum': torch.zeros((embedding_size), device=device),
			'count': torch.tensor(0.0, device=device)
		}
		for cls in range(args.num_classes)
	}

	# Process batches of images and embeddings
	for batch_idx, (images, embeddings, target) in progress_bar:
		images = images.to(device, non_blocking=True)
		embeddings = embeddings.to(device, non_blocking=True)
		target = target.to(device, non_blocking=True)

		# Flatten the images to the shape (batch_size, image_size)
		images_flattened = images.view(images.size(0), -1)

		# Update centroids and anti-centroids for both image data and embeddings
		centroids_images, anti_centroids_images = update_local_centroids_and_anti_centroids(
			centroids_images, anti_centroids_images, images_flattened, target, args.num_classes
		)
		centroids_embeddings, anti_centroids_embeddings = update_local_centroids_and_anti_centroids(
			centroids_embeddings, anti_centroids_embeddings, embeddings, target, args.num_classes
		)

	# Aggregate centroids and anti-centroids across all GPUs for image data
	centroids_images, anti_centroids_images = all_reduce_centroids_and_anti_centroids(
		centroids_images, anti_centroids_images, args.num_classes
	)
	# Aggregate centroids and anti-centroids across all GPUs for embeddings
	centroids_embeddings, anti_centroids_embeddings = all_reduce_centroids_and_anti_centroids(
		centroids_embeddings, anti_centroids_embeddings, args.num_classes
	)

	# Compute final centroids and anti-centroids for image data
	final_centroids_images, final_anti_centroids_images = compute_final_centroids_and_anti_centroids(
		centroids_images, anti_centroids_images, args.num_classes, image_size
	)

	# Compute final centroids and anti-centroids for embeddings
	final_centroids_embeddings, final_anti_centroids_embeddings = compute_final_centroids_and_anti_centroids(
		centroids_embeddings, anti_centroids_embeddings, args.num_classes, embedding_size
	)

	# Normalize the centroids and anti-centroids
	final_centroids_images = F.normalize(final_centroids_images, p=2, dim=1)
	final_anti_centroids_images = F.normalize(final_anti_centroids_images, p=2, dim=1)
	final_centroids_embeddings = F.normalize(final_centroids_embeddings, p=2, dim=1)
	final_anti_centroids_embeddings = F.normalize(final_anti_centroids_embeddings, p=2, dim=1)

	return final_centroids_images, final_anti_centroids_images, final_centroids_embeddings, final_anti_centroids_embeddings

def get_counts(cls, features, labels, centroids, anti_centroids, counts, anti_counts):
		# Calculate centroids where the label is 1
		positive_mask = labels[:, cls] == 1.
		if positive_mask.sum() > 0:
			# Sum of features for the positive class
			if features.dim() == 2:
				centroids[cls] = features[positive_mask, :].sum(dim=0)
			else:
				centroids[cls] = features[positive_mask, cls, :].sum(dim=0)
			counts[cls] = positive_mask.sum()

		# Calculate anti-centroids where the label is 0
		negative_mask = labels[:, cls] == 0.
		if negative_mask.sum() > 0:
			# Sum of features for the negative class
			if features.dim() == 2:
				anti_centroids[cls] = features[negative_mask,:].sum(dim=0)
			else:
				anti_centroids[cls] = features[negative_mask, cls, :].sum(dim=0)
			anti_counts[cls] = negative_mask.sum()

		return centroids, anti_centroids, counts, anti_counts

def compute_centroids(features, labels, num_classes):
	centroids = torch.zeros((num_classes, features.shape[-1]), device=features.device)
	anti_centroids = torch.zeros((num_classes, features.shape[-1]), device=features.device)
	counts = torch.zeros(num_classes, device=features.device)
	anti_counts = torch.zeros(num_classes, device=features.device)

	centroids, anti_centroids, counts, anti_counts = get_counts(0, features, labels, centroids, anti_centroids, counts, anti_counts)
	centroids, anti_centroids, counts, anti_counts = get_counts(1, features, labels, centroids, anti_centroids, counts, anti_counts)
	centroids, anti_centroids, counts, anti_counts = get_counts(2, features, labels, centroids, anti_centroids, counts, anti_counts)
	centroids, anti_centroids, counts, anti_counts = get_counts(3, features, labels, centroids, anti_centroids, counts, anti_counts)
	centroids, anti_centroids, counts, anti_counts = get_counts(4, features, labels, centroids, anti_centroids, counts, anti_counts)

	# Reduce centroids and counts across all processes (all GPUs)
	dist.all_reduce(centroids, op=dist.ReduceOp.SUM)
	dist.all_reduce(counts, op=dist.ReduceOp.SUM)

	# Reduce anti-centroids and anti-counts across all processes (all GPUs)
	dist.all_reduce(anti_centroids, op=dist.ReduceOp.SUM)
	dist.all_reduce(anti_counts, op=dist.ReduceOp.SUM)

	# Avoid division by zero and compute the centroids
	counts = counts.clamp(min=1.0)  # Ensure no division by zero
	centroids /= counts.unsqueeze(1)

	anti_counts = anti_counts.clamp(min=1.0)  # Ensure no division by zero
	anti_centroids /= anti_counts.unsqueeze(1)

	# Normalize the centroids and anti-centroids
	centroids = F.normalize(centroids, p=2, dim=1)
	anti_centroids = F.normalize(anti_centroids, p=2, dim=1)

	return centroids, anti_centroids

def get_IB_Loss(input_scores, feature_scores_first, features_scores_second, Q):
	term_1 = torch.sum(feature_scores_first.unsqueeze(2) * input_scores.unsqueeze(1) * torch.log(input_scores.unsqueeze(1) + 1e-9),dim=(-2,-1))
	term_2 = torch.sum(feature_scores_first.unsqueeze(2) * features_scores_second.unsqueeze(1) * torch.log(Q.unsqueeze(0) + 1e-9),dim=(-2,-1))
	bound = term_1 - term_2
	return bound.mean()

def compute_cluster_scores(feat, centroids, device, beta=1.0):
	N = feat.shape[0]
	K = centroids.shape[0]
	score = torch.zeros(N, K).to(device)
	feat = F.normalize(feat, p=2, dim=1)
	score = torch.cdist(feat, centroids, p=2) ** 2
	score =  - beta * score
	score = torch.softmax(score, dim=-1)
	return score

def compute_input_scores(dataloader, image_centers, image_anti_centers, embedding_centers, embedding_anti_centers, device, rank, args):
	if rank == 0:
		progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Compute Input scores", position=0)
	else:
		progress_bar = enumerate(dataloader)

	inputs_scores = torch.empty((0, )).to(device)
	embeddings_scores = torch.empty((0, )).to(device)

	for batch_idx, (input, embedding, targets) in progress_bar:
		inter_score = torch.empty((0, )).to(device)
		for cls in range(args.num_classes):
			image_centers_cls = image_centers[cls].unsqueeze(0)
			image_anti_centers_cls = image_anti_centers[cls].unsqueeze(0)
			cluster_centers = torch.cat([image_centers_cls, image_anti_centers_cls], dim=0)
			input_score = compute_cluster_scores(input.view(input.shape[0], -1), cluster_centers, device) # (N, 2)
			input_score = input_score.unsqueeze(1)
			inter_score = torch.cat((inter_score, input_score), dim=1)
		inputs_scores = torch.cat((inputs_scores, inter_score), dim=0)

		inter_score = torch.empty((0, )).to(device)
		for cls in range(args.num_classes):
			embedding_centers_cls = embedding_centers[cls].unsqueeze(0)
			embedding_anti_centers_cls = embedding_anti_centers[cls].unsqueeze(0)
			cluster_centers = torch.cat([embedding_centers_cls, embedding_anti_centers_cls], dim=0)
			input_score = compute_cluster_scores(embedding, cluster_centers, device) # (N, 2)
			input_score = input_score.unsqueeze(1)
			inter_score = torch.cat((inter_score, input_score), dim=1)
		embeddings_scores = torch.cat((embeddings_scores, inter_score), dim=0)

	return inputs_scores, embeddings_scores

@torch.no_grad()
def compute_svd(features, indexes, device):
	n, dim = features.shape[0], features.shape[1]
	U, V = torch.empty((n, dim)).cuda(), torch.empty((dim, dim)).cuda()

	sorted_indexes = torch.argsort(indexes)
	features = features[sorted_indexes].cpu().numpy()

	U, _, V = scipy.linalg.svd(features, full_matrices=False)
	U, V = torch.tensor(U).cuda().contiguous(), torch.tensor(V).cuda().contiguous()
	del features, indexes
	torch.cuda.empty_cache()
	U, V = U.to(device), V.to(device)
	return U, V
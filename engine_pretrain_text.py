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

def train_one_epoch(text_model, dataloader, criterion, optimizer, scaler, device, rank, epoch, num_epochs, args):
	text_model.train()
	epoch_loss = []
	optimizer.zero_grad()

	if rank == 0:
		progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", position=0)
	else:
		progress_bar = enumerate(dataloader)

	# Inference loop
	for batch_idx, (_, embedding, targets, _, _, _) in progress_bar:
		adjust_learning_rate(optimizer, batch_idx / len(dataloader) + epoch, args)

		embedding = embedding.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		# Mixed precision training with gradient scaler
		with torch.cuda.amp.autocast(dtype=torch.bfloat16):
			text_logits, _ = text_model(embedding)

			# Compute cross-entropy loss for this batch and accumulate
			loss = criterion(text_logits, targets)

		# Scale and backpropagate the final loss
		scaler(loss, optimizer, clip_grad=None, parameters=text_model.parameters(), create_graph=False, update_grad=True)

		epoch_loss.append(loss.item())

		optimizer.zero_grad()
		torch.cuda.synchronize()

		if rank == 0:
			progress_bar.set_postfix(loss=loss.item())

	k = len(epoch_loss)
	return sum(epoch_loss) / k
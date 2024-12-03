import json
import os
import torch
from torch.utils.data import Dataset
from timm.data import create_transform
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


# Custom dataset class
class MedicalReportDataset(Dataset):
    def __init__(self, json_file):
        # Load data from the JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # Convert NaN in findings to empty strings
        for item in self.data:
            if pd.isna(item["section_findings"]):
                item["section_findings"] = ""

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get the specific data item
        item = self.data[idx]

        path = item["Path"]
        # Format the report: "findings: section_findings, impression: section_impression"
        report = f"findings: {item['section_findings']}, impression: {item['section_impression']}"

        # Extract the labels (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion)
        labels = torch.tensor([
            item["Cardiomegaly"],
            item["Edema"],
            item["Consolidation"],
            item["Atelectasis"],
            item["Pleural Effusion"]
        ], dtype=torch.float32)

        return path, report, labels

# Custom dataset class
class CheXpertVLMDataset(Dataset):
	def __init__(self, json_file, embedding_file, logits_file, image_root, image_centers, image_anti_centers, embedding_centers, embedding_anti_centers, transform, mode='train'):
		self.image_root = image_root
		self.mode = mode
		self.transform = transform
		if mode == 'train':
			self.embedding_dict = self.get_embedding_dict(embedding_file)
			self.logits_dict = self.get_text_logits(logits_file)
		
			self.image_centers, self.image_anti_centers, self.embedding_centers, self.embedding_anti_centers = image_centers, image_anti_centers, embedding_centers, embedding_anti_centers

		# Load data from the JSON file
		with open(json_file, 'r') as f:
			self.data = json.load(f)
		
		# Convert NaN in findings to empty strings
		for item in self.data:
			if pd.isna(item["section_findings"]):
				item["section_findings"] = ""

	def get_embedding_dict(self, embedding_file):
		embedding_dict = torch.load(embedding_file, map_location='cpu')
		return embedding_dict
	
	def get_text_logits(self, logits_file):
		logits_dict = torch.load(logits_file, map_location='cpu')
		return logits_dict
	
	def __len__(self):
		# Return the total number of samples
		return len(self.data)

	def compute_cluster_scores(self, feat, centroids, beta=1.0):
		N = feat.shape[0]
		K = centroids.shape[0]
		score = torch.zeros(N, K)
		feat = F.normalize(feat, p=2, dim=1)
		score = torch.cdist(feat, centroids, p=2) ** 2
		score =  - beta * score
		score = torch.softmax(score, dim=-1)
		return score

	def compute_input_score(self, image, embedding):
		image_score = torch.empty((0, ))
		for cls in range(5):
			image_centers_cls = self.image_centers[cls].unsqueeze(0)
			image_anti_centers_cls = self.image_anti_centers[cls].unsqueeze(0)
			cluster_centers = torch.cat([image_centers_cls, image_anti_centers_cls], dim=0)
			input_score = self.compute_cluster_scores(image.view(-1).unsqueeze(0), cluster_centers) # (N, 2)
			input_score = input_score.unsqueeze(1)
			image_score = torch.cat((image_score, input_score), dim=1)

		embedding_score = torch.empty((0, ))
		for cls in range(5):
			embedding_centers_cls = self.embedding_centers[cls].unsqueeze(0)
			embedding_anti_centers_cls = self.embedding_anti_centers[cls].unsqueeze(0)
			cluster_centers = torch.cat([embedding_centers_cls, embedding_anti_centers_cls], dim=0)
			input_score = self.compute_cluster_scores(embedding.unsqueeze(0), cluster_centers) # (N, 2)
			input_score = input_score.unsqueeze(1)
			embedding_score = torch.cat((embedding_score, input_score), dim=1)
		
		return image_score.squeeze(0), embedding_score.squeeze(0)

	def __getitem__(self, idx):
		# Get the specific data item
		item = self.data[idx]
		image_path = os.path.join(self.image_root, item["Path"])
		image = Image.open(image_path).convert('RGB')
		image = self.transform(image)

		# Extract the labels (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion)
		labels = torch.tensor([
			item["Cardiomegaly"],
			item["Edema"],
			item["Consolidation"],
			item["Atelectasis"],
			item["Pleural Effusion"]
		], dtype=torch.float32)

		if self.mode == 'train':
			embedding = self.embedding_dict[item["Path"]]['embedding']
			text_logits = self.logits_dict[item["Path"]]
			image_score, embedding_score = self.compute_input_score(image, embedding)
			return image, embedding, labels, image_score, embedding_score, text_logits, idx
		
		return image, labels

def build_transform(is_train, args):
	mean, std = (0.5056, 0.5056, 0.5056), (0.252, 0.252, 0.252)

	if is_train:
		return create_transform(
			input_size=args.input_size,
			is_training=True,
			color_jitter=None,
			auto_augment='rand-m6-mstd0.5-inc1',
			interpolation='bicubic',
			re_prob=0.25,
			re_mode='pixel',
			re_count=1,
			mean=mean,
			std=std,
		)
	
	size = int(args.input_size / (224 / 256 if args.input_size <= 224 else 1.0))
	return transforms.Compose([
		transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
		transforms.CenterCrop(args.input_size),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])

def build_dataset_chest_xray(split, image_centers, image_anti_centers, embedding_centers, embedding_anti_centers, args):
	is_train = (split == 'train')
	transform = build_transform(is_train, args)
	file = args.train_file if is_train else args.test_file
	embedding_file = args.train_embedding_file
	logits_file = args.train_logits_file
	mode = 'train' if is_train else 'test'
	dataset = CheXpertVLMDataset(json_file=file, embedding_file=embedding_file, logits_file=logits_file, image_root=args.data_path, image_centers=image_centers, 
							  image_anti_centers=image_anti_centers, embedding_centers=embedding_centers, embedding_anti_centers=embedding_anti_centers, transform=transform, mode=mode)
	return dataset
import torch
from transformers import AutoModel
import torch.nn.functional as F
from dataloader import MedicalReportDataset
from tqdm import tqdm
import os

# Save passage embeddings
def save_embeddings_and_labels(file_paths, embeddings, labels, save_path):
    # Create a dictionary where the file_path is the key and a sub-dictionary with embedding and label is the value
    data_dict = {file_paths[i]: {'embedding': embeddings[i], 'label': labels[i]} for i in range(len(file_paths))}
    torch.save(data_dict, save_path)
    print(f"Embeddings, labels, and file paths saved to {save_path}")

# Main function to run inference on a single GPU
def main(device, json_path, save_path, max_length=2048):
    # Set the device (use 'cuda' if available, otherwise fall back to CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and data loader
    dataset = MedicalReportDataset(json_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

    # Load the model
    model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True).to(device, dtype=torch.bfloat16)
    model.eval()

    # Initialize empty lists for embeddings, labels, and file paths
    passage_embeddings_list = []
    labels_list = []
    file_paths_list = []  # New list to store image paths

    # Create the tqdm progress bar
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference")

    # Inference loop
    for batch_idx, batch in progress_bar:
        # Assuming the dataset returns file paths, passages, and labels in the batch
        file_paths = batch[0]  # Assuming batch[0] contains file paths
        passages = batch[1]    # Modify if needed according to your dataset
        labels = batch[2].to(device)

        # Encode and normalize the passage embeddings with mixed precision
        with torch.no_grad():
            passage_embeddings = model.encode(passages, instruction="", max_length=max_length)
            passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

        # Append embeddings, labels, and file paths to the lists
        passage_embeddings_list.append(passage_embeddings)
        labels_list.append(labels)
        file_paths_list.extend(file_paths)  # Extend, not append, since it's a list of paths

        # Get current GPU memory usage (for each GPU)
        gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
        gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # Convert to MB

        # Update the progress bar description with current batch info and GPU memory usage
        progress_bar.set_postfix(batch=batch_idx + 1, gpu_mem_alloc=f'{gpu_memory_allocated:.2f} MB', gpu_mem_res=f'{gpu_memory_reserved:.2f} MB')

    # Concatenate embeddings and labels from all batches
    passage_embeddings_tensor = torch.cat(passage_embeddings_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)

    # Save the gathered embeddings, labels, and file paths
    save_embeddings_and_labels(file_paths_list, passage_embeddings_tensor, labels_tensor, save_path)

if __name__ == "__main__":
    json_path = '/home/local/ASURITE/rgoel15/ICML_VLM/chexpert_report.json'  # Path to the JSON file
    save_path = '/home/local/ASURITE/rgoel15/ICML_VLM/passage_embeddings_1.pt'  # Path to save the embeddings

    # Run on a single GPU or CPU
    main('cuda', json_path, save_path)

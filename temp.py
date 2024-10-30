@torch.no_grad()
def get_image_centers(dataloader, device, rank, args):
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
        for cls in range(args.nb_classes)
    }
    anti_centroids_images = {
        cls: {
            'sum': torch.zeros((image_size), device=device),
            'count': torch.tensor(0.0, device=device)
        }
        for cls in range(args.nb_classes)
    }

    centroids_embeddings = {
        cls: {
            'sum': torch.zeros((embedding_size), device=device),
            'count': torch.tensor(0.0, device=device)
        }
        for cls in range(args.nb_classes)
    }
    anti_centroids_embeddings = {
        cls: {
            'sum': torch.zeros((embedding_size), device=device),
            'count': torch.tensor(0.0, device=device)
        }
        for cls in range(args.nb_classes)
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
            centroids_images, anti_centroids_images, images_flattened, target, args.nb_classes
        )
        centroids_embeddings, anti_centroids_embeddings = update_local_centroids_and_anti_centroids(
            centroids_embeddings, anti_centroids_embeddings, embeddings, target, args.nb_classes
        )

    # Aggregate centroids and anti-centroids across all GPUs for image data
    centroids_images, anti_centroids_images = all_reduce_centroids_and_anti_centroids(
        centroids_images, anti_centroids_images, args.nb_classes
    )
    # Aggregate centroids and anti-centroids across all GPUs for embeddings
    centroids_embeddings, anti_centroids_embeddings = all_reduce_centroids_and_anti_centroids(
        centroids_embeddings, anti_centroids_embeddings, args.nb_classes
    )

    # Compute final centroids and anti-centroids for image data
    final_centroids_images, final_anti_centroids_images = compute_final_centroids_and_anti_centroids(
        centroids_images, anti_centroids_images, args.nb_classes, image_size
    )

    # Compute final centroids and anti-centroids for embeddings
    final_centroids_embeddings, final_anti_centroids_embeddings = compute_final_centroids_and_anti_centroids(
        centroids_embeddings, anti_centroids_embeddings, args.nb_classes, embedding_size
    )

    # Normalize the centroids and anti-centroids
    final_centroids_images = F.normalize(final_centroids_images, p=2, dim=1)
    final_anti_centroids_images = F.normalize(final_anti_centroids_images, p=2, dim=1)
    final_centroids_embeddings = F.normalize(final_centroids_embeddings, p=2, dim=1)
    final_anti_centroids_embeddings = F.normalize(final_anti_centroids_embeddings, p=2, dim=1)

    return final_centroids_images, final_anti_centroids_images, final_centroids_embeddings, final_anti_centroids_embeddings

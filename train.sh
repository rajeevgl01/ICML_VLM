#!/bin/bash

torchrun --nproc_per_node=2 main_finetune.py \
--train_file chexpert_report.json --test_file chexpert_report_test.json \
--train_embedding_file passage_embeddings_1.pt --data_path /home/local/ASURITE/rgoel15/CheXpert-v1.0-small \
--finetune /home/local/ASURITE/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--finetune_adapter run_2024-10-14_14-27-30/best_auc.pt \
--layer_decay 0.4 --text_ib_weight 0

torchrun --nproc_per_node=2 main_finetune.py \
--train_file chexpert_report.json --test_file chexpert_report_test.json \
--train_embedding_file passage_embeddings_1.pt --data_path /home/local/ASURITE/rgoel15/CheXpert-v1.0-small \
--finetune /home/local/ASURITE/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--finetune_adapter run_2024-10-14_14-27-30/best_auc.pt \
--layer_decay 0.4 --text_ib_weight 5

torchrun --nproc_per_node=2 main_finetune.py \
--train_file chexpert_report.json --test_file chexpert_report_test.json \
--train_embedding_file passage_embeddings_1.pt --data_path /home/local/ASURITE/rgoel15/CheXpert-v1.0-small \
--finetune /home/local/ASURITE/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--finetune_adapter run_2024-10-14_14-27-30/best_auc.pt \
--layer_decay 0.4 --text_ib_weight 20

torchrun --nproc_per_node=2 main_finetune.py \
--train_file chexpert_report.json --test_file chexpert_report_test.json \
--train_embedding_file passage_embeddings_1.pt --data_path /home/local/ASURITE/rgoel15/CheXpert-v1.0-small \
--finetune /home/local/ASURITE/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--finetune_adapter run_2024-10-14_14-27-30/best_auc.pt \
--layer_decay 0.4 --text_ib_weight 40

torchrun --nproc_per_node=2 main_finetune.py \
--train_file chexpert_report.json --test_file chexpert_report_test.json \
--train_embedding_file passage_embeddings_1.pt --data_path /home/local/ASURITE/rgoel15/CheXpert-v1.0-small \
--finetune /home/local/ASURITE/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--finetune_adapter run_2024-10-14_14-27-30/best_auc.pt \
--layer_decay 0.4 --text_ib_weight 50

torchrun --nproc_per_node=2 main_finetune.py \
--train_file chexpert_report.json --test_file chexpert_report_test.json \
--train_embedding_file passage_embeddings_1.pt --data_path /home/local/ASURITE/rgoel15/CheXpert-v1.0-small \
--finetune /home/local/ASURITE/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--finetune_adapter run_2024-10-14_14-27-30/best_auc.pt \
--layer_decay 0.4 --text_ib_weight 100
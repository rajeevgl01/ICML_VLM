#!/bin/bash

# torchrun --nproc_per_node=2 main_finetune_final_mfa.py \
# --train_file chexpert_report.json --test_file chexpert_report_test.json \
# --train_embedding_file passage_embeddings_1.pt --data_path /home/local/ASURITE/rgoel15/CheXpert-v1.0-small \
# --train_logits_file passage_logits.pt \
# --finetune /home/local/ASURITE/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
# --finetune_adapter run_2024-10-14_14-27-30/best_auc.pt \
# --layer_decay 0.4 --soft_rank 38 --tn_loss_weight 0.000625 --num_epochs 75

torchrun --nproc_per_node=2 main_finetune_final_mfa.py \
--train_file chexpert_report.json --test_file chexpert_report_test.json \
--train_embedding_file passage_embeddings_1.pt --data_path /home/local/ASURITE/rgoel15/CheXpert-v1.0-small \
--train_logits_file passage_logits.pt \
--retrain /home/local/ASURITE/rgoel15/ICML_VLM/mfa2/checkpoints/run_2024-10-31_01-11-19/best_auc.pt \
--layer_decay 0.4 --soft_rank 38 --tn_loss_weight 0.000625 --num_epochs 75
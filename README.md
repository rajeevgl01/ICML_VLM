# Installation

Install the environment using the 'vlm.yaml' file.

# Text Data processing and backbone training

## process embedding

1. change path of json medical report at line 68
Note: Files should be in this format
```json
[
    {
        "path_to_image": "valid/patient64620/study1/view1_frontal.jpg",
        "section_findings": NaN,
        "section_impression": "\n1. SINGLE VIEW OF CHEST DEMONSTRATES NO SIGNIFICANT INTERVAL\nCHANGE IN POSITION OF BILATERAL CHEST TUBES. MEDIAN STERNOTOMY\nWIRES ARE UNCHANGED.\n2. PREVIOUSLY DESCRIBED LEFT APICAL PNEUMOTHORAX IS NOT WELL SEEN\nON TODAY'S STUDY. NO LARGE PNEUMOTHORAX IS IDENTIFIED.\n3. THERE HAS BEEN INTERVAL INCREASE IN THE LEFT PLEURAL EFFUSION\nWITH PERSISTENT RETROCARDIAC ATELECTASIS. SMALL RIGHT PLEURAL\nEFFUSION REMAINS.\n",
        "Path": "valid/patient64620/study1/view1_frontal.jpg",
        "Cardiomegaly": 0.0,
        "Edema": 0.0,
        "Consolidation": 0.0,
        "Atelectasis": 1.0,
        "Pleural Effusion": 1.0
    },
    {
        "path_to_image": "valid/patient64678/study1/view1_frontal.jpg",
        "section_findings": NaN,
        "section_impression": "\n1. INTERVAL REMOVAL OF PREVIOUSLY SEEN SKIN STAPLES.\n2. STABLE APPEARANCE OF THE HEART AND LUNGS, WITH NO SIGNIFICANT\nINTERVAL CHANGE IN PULMONARY EDEMA AND RETROCARDIAC LEFT LOWER LOBE\nATELECTASIS, CONSOLIDATION AND/OR EFFUSION.\n",
        "Path": "valid/patient64678/study1/view1_frontal.jpg",
        "Cardiomegaly": 0.0,
        "Edema": 0.0,
        "Consolidation": 0.0,
        "Atelectasis": 1.0,
        "Pleural Effusion": 1.0
    },.....
]
```
2. change path to save file at line 69

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 obtain_embeddings.py
```

## pretrain text backbone
Note: read from line 119 in pretrain_text_backbone.py to know what each argument does

```
torchrun --nproc_per_node=4 pretrain_text_backbone.py \
--train_file \path\to\file \
--test_file \path\to\file \
--checkpoint_root_dir \path\to\dir \
--num_classes 5
```

# Train MFA model
Note: read from line 212 in main_finetune.py to know what each argument does
```
torchrun --nproc_per_node=4 main_finetune.py \
--train_file \path\to\file \
--test_file \path\to\file \
--train_embedding_file \path\to\file \
--checkpoint_root_dir \path\to\dir \
--num_classes 5
```
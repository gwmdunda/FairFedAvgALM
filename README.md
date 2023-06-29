# Handling Group Fairness in Federated Learning Using Augmented Lagrangian Approach


This repo contains the code of our paper Handling Group Fairness in Federated Learning Using Augmented Lagrangian Approach.


> [Handling Group Fairness in Federated Learning Using
Augmented Lagrangian Approach](https://openreview.net/forum?id=h4y9gvhB-V)

<!-- <img src=""  width="897" height="317" /> -->

## Requirements
- pytorch 
- numpy
- scikit-learn
- timm
- pandas
- filelock
- ray tune
- pyyaml
- tensorboard

## Configuration

The hyper-parameters and the training options are defined in config.yaml. Please refer to Appendix C for the complete list of hyperparameters used in the paper.

## Datasets

The user can set arbitrary root folder for dataset [dataset_dir] (dataset folder in this repo for the skeleton). Inside [dataset_dir], we have the following essential directory structure for CelebA and imsitu datasets.
```
[dataset_dir]
├── CelebA
└── imsitu
```
The CelebA directory structure is
```
CelebA
├── celeba-attractive-test.csv
├── celeba-attractive-train.csv
├── celeba-attractive-valid.csv
├── img_align_celeba
├── list_attr_celeba.txt
└── list_eval_partition.txt
```
image contents inside img_align_celeba can be downloaded from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ).

The imsitu directory structure is
```
imsitu
├── metadata
│   ├── test.data
│   ├── test_ratio_1.ids
│   ├── train.data
│   ├── train_ratio_1.ids
│   ├── val.data
│   ├── val_ratio_1.ids
│   └── verb_id.map
│
└── of500_images_resized
```
image contents inside of500_images_resized can be downloaded from [here](https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar)
## Usage

The code covers:
- FL training environment
- Federated optimization for the proposed method and baselines
- The results of the experiment such as performance metrics as function of communication rounds and the final trained model 

Navigate to the project directory and run the following to start the experiment
```bash
python main.py
```
The state and the result of the experiment is stored in ./ray_results/[experiment_name]. Run the following to visualize the performance metrics result
```bash
tensorboard --logdir ray_results/[experiment_name]
```
The final trained model can be found in ./ray_results/[experiment_name]/train_instance_xxx/model_checkpoint. 


## Citation

If you find our work useful in your research, please consider citing:

```latex

```


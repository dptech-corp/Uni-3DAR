Uni-3DAR
========
[[Paper](https://arxiv.org/abs/2503.16278)]

Introduction
------------

<p align="center"><img src="fig/overview.png" width=95%></p>
<p align="center"><b>Schematic illustration of the Uni-3DAR framework</b></p>

Uni-3DAR is an autoregressive model that unifies various 3D tasks. In particular, it offers the following improvements:

1. **Unified Handling of Multiple 3D Data Types.**  
   Although we currently focus on microscopic structures such as molecules, proteins, and crystals, the proposed method can be seamlessly applied to macroscopic 3D structures.

2. **Support for Diverse Tasks.**  
   Uni-3DAR naturally supports a wide range of tasks within a single model, especially for both generation and understanding.

3. **High Efficiency.**  
   It uses octree compression-in combination with our proposed 2-level subtree compression-to represent the full 3D space using only hundreds of tokens, compared with tens of thousands in a full-size grid. Our inference benchmarks also show that Uni-3DAR is much faster than diffusion-based models.

4. **High Accuracy.**  
   Building on octree compression, Uni-3DAR further tokenizes fine-grained 3D patches to maintain structural details, achieving substantially better generation quality than previous diffusion-based models.


News
----

**2025-04-09:** We have released the core model along with the MP20 (Crystal) training and inference pipeline.

**2025-04-07:** We have released the core model along with the DRUG training and inference pipeline.

**2025-03-21:** We have released the core model along with the QM9 training and inference pipeline.


Dependencies
------------

- [Uni-Core](https://github.com/dptech-corp/Uni-Core). For convenience, you can use our prebuilt Docker image:  
  `docker pull dptechnology/unicore:2407-pytorch2.4.0-cuda12.5-rdma`


Reproducing Results on QM9
--------------------------

To reproduce results on the QM9 dataset using our pretrained model or train from scratch, please follow the instructions below.

### Download Pretrained Model and Dataset

Download the pretrained checkpoint (`qm9.pt`) and the dataset archive (`qm9_data.tar.gz`) from our [Hugging Face repository](https://huggingface.co/dptech/Uni-3DAR/tree/main).

### Inference with Pretrained Model

To generate QM9 molecules using the pretrained model:

```
bash scripts/inference_qm9.sh qm9.pt
```

### Train from Scratch

To train the model from scratch:

1. Extract the dataset:
```
tar -xzvf qm9_data.tar.gz
```

2. Run the training script with your desired data path and experiment name:

```
base_dir=/your_folder_to_save/ batch_size=16 bash scripts/train_qm9.sh ./qm9_data/ name_of_your_exp
```

Note: By default, we train QM9 using 4 GPUs, with a total batch size of `4 × 16 = 64`. You may adjust the batch size based on your available GPU configuration.


Reproducing Results on DRUG
---------------------------

To reproduce results on the DRUG dataset using our pretrained model or train from scratch, please follow the instructions below.

### Download Pretrained Model and Dataset

Download the pretrained checkpoint (`drug.pt`) and the dataset archive (`drug_data.tar.gz`) from our [Hugging Face repository](https://huggingface.co/dptech/Uni-3DAR/tree/main).

### Inference with Pretrained Model

To generate DRUG molecules using the pretrained model:

```
bash scripts/inference_drug.sh drug.pt
```

### Train from Scratch

To train the model from scratch:

1. Extract the dataset:
```
tar -xzvf drug_data.tar.gz
```

2. Run the training script with your desired data path and experiment name:

```
base_dir=/your_folder_to_save/ batch_size=16 bash scripts/train_drug.sh ./drug_data/ name_of_your_exp
```

Note: By default, we train DRUG using 8 GPUs, with a total batch size of `8 × 16 = 128`. You may adjust the batch size based on your available GPU configuration.


Reproducing Results on MP20 (Crystal)
-------------------------------------

To reproduce results on the MP20 dataset using our pretrained model or train from scratch, please follow the instructions below.

### Download Pretrained Model and Dataset

Download the pretrained checkpoint (`mp20.pt`, `mp20_csp.pt` and `mp20_pxrd.pt`) and the dataset archive (`mp20_data.tar.gz`) from our [Hugging Face repository](https://huggingface.co/dptech/Uni-3DAR/tree/main). 

First, you should extract the dataset, which will be used in both training and evaluation:

```
tar -xzvf mp20_data.tar.gz
```

### Inference with Pretrained Model

For de-novo MP20 crystal generation:

```
bash scripts/inference_mp20.sh mp20.pt ./mp20_data/test.csv
```

For MP20 crystal structure prediction (CSP):

```
data_path=./mp20_data/ bash scripts/inference_mp20_csp.sh mp20_csp.pt
```

For MP20 PXRD-guided CSP:

```
data_path=./mp20_data/ bash scripts/inference_mp20_pxrd.sh mp20_pxrd.pt
```

### Train from Scratch


For de-novo MP20 crystal generation training:

```
base_dir=/your_folder_to_save/ batch_size=16 bash scripts/train_mp20.sh ./mp20_data/ name_of_your_exp
```

Note: By default, we use 4 GPUs, with a total batch size of `4 × 16 = 64`. You may adjust the batch size based on your available GPU configuration.


For MP20 CSP training:

```
base_dir=/your_folder_to_save/ batch_size=8 bash scripts/train_mp20_csp.sh ./mp20_data/ name_of_your_exp
```

Note: By default, we use 8 GPUs, with a total batch size of `8 × 8 = 64`. You may adjust the batch size based on your available GPU configuration.


For MP20 PRXD-guided CSP training:

```
base_dir=/your_folder_to_save/ batch_size=8 bash scripts/train_mp20_pxrd.sh ./mp20_data/ name_of_your_exp
```

Note: By default, we use 8 GPUs, with a total batch size of `8 × 8 = 64`. You may adjust the batch size based on your available GPU configuration.


Reproducing Results on Molecular Property Prediction
-------------------------------------

To reproduce results on the molecular property prediction downstream tasks using our pretrained model, please follow the instructions below.

### Download Pretrained Model and Dataset

Download the pretrained checkpoint (`mol_pretrain_weight.pt`) and the dataset archive (`mol_downstream.tar.gz`) from our [Hugging Face repository](https://huggingface.co/dptech/Uni-3DAR/tree/main). 

First, you should extract the dataset, which will be used in both training and evaluation:

```
tar -xzvf mol_downstream.tar.gz
```

### Finetune with Pretrained Model

For HOMO, LUMO and GAP:

```
task=homo # or lumo, gap
bash scripts/train_rep_downstream.sh $dataset_path/scaffold_ood_qm9 $your_folder_to_save $task mol_pretrain_weight.pt "5e-5 1e-4" "32 64" "200" "0.0" "0.06" "0 1 2" 
```

For E1-CC2, E2-CC2, f1-CC2 and f2-CC2:
 
```
task=E1-CC2 # or E2-CC2, f1-CC2, f2-CC2
bash scripts/train_rep_downstream.sh $dataset_path/scaffold_ood_qm8 $your_folder_to_save $task mol_pretrain_weight.pt "5e-5 1e-4" "32 64" "200" "0.0" "0.06" "0 1 2" 
```

For Dipmom, aIP and D3 Dispersion Corrections:
```
task=Dipmom_Debye # or aIP_eV, D3_disp_corr_eV
bash scripts/train_rep_downstream.sh $dataset_path/scaffold_ood_compas1d_cam $your_folder_to_save $task mol_pretrain_weight.pt "5e-5 1e-4" "32 64" "200" "0.0" "0.06" "0 1 2" 
```

Note: By default, we use 1 GPU for finetune.



Reproducing Results on Protein Pocket Prediction
-------------------------------------

To reproduce results on the protein pocket prediction downstream tasks using our pretrained model, please follow the instructions below.

### Download Pretrained Model and Dataset

Download the pretrained checkpoint (`protein_pretrain_weight.pt`) and the dataset archive (`protein_downstream.tar.gz`) from our [Hugging Face repository](https://huggingface.co/dptech/Uni-3DAR/tree/main). 

First, you should extract the dataset, which will be used in both training and evaluation:

```
tar -xzvf protein_downstream.tar.gz
```

### Finetune with Pretrained Model


```
task=binding
bash scripts/train_rep_downstream.sh $dataset_path/binding $your_folder_to_save $task protein_pretrain_weight.pt "5e-5 1e-4" "32 64" "100" "0.1" "0.06" "0" 
```

Note: By default, we use 8 GPUs for finetune. You may adjust the batch size based on your available GPU configuration.



Citation
--------

Please kindly cite our papers if you use the data/code/model.

```
@article{lu2025uni3dar,
  author    = {Shuqi Lu and Haowei Lin and Lin Yao and Zhifeng Gao and Xiaohong Ji and Weinan E and Linfeng Zhang and Guolin Ke},
  title     = {Uni-3DAR: Unified 3D Generation and Understanding via Autoregression on Compressed Spatial Tokens},
  journal   = {Arxiv},
  year      = {2025},
}
```




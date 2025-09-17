# GenView++: Adaptive View Generation and Quality-Driven Supervision for Vision and Vision-Language Representation Learning

![GenView++ Framework](figs/framework.png)

This repository contains the official implementation of **GenView++: Adaptive View Generation and Quality-Driven Supervision for Vision and Vision-Language Representation Learning**, presented at ECCV 2024.

> **[GenView++: Adaptive View Generation and Quality-Driven Supervision for Vision and Vision-Language Representation Learning](https://arxiv.org/abs/xxx)**<br> 

## 🔨 Installation

Follow the steps below to set up the environment and install dependencies.

### Step 1: Create and Activate a Conda Environment

Create a new Conda environment with Python 3.8 and activate it:

```bash
conda create --name env_genview python=3.8 -y
conda activate env_genview
```

### Step 2: Install Required Packages

You can install PyTorch, torchvision, and torchaudio via pip or Conda. Choose the command based on your preference and GPU compatibility.

```bash
# Using pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Or using conda
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Step 3: Clone the Repository and Install Project Dependencies

Clone the GenView repository and install the required dependencies:

```bash
git clone https://github.com/xiaojieli0903/GenViewPlusPlus.git
cd GenViewPlusPlus
pip install -r requirements.txt
```

Apply modifications to `open_clip` and `timm`:

```bash
sh tools/toolbox_genview/change_openclip_timm.sh
```

## 📷 Adaptive View Generation

### **Step 1: calculate the noise level and guidance scale of each original image**

Please refer to [data_generation/README.md](data_generation/README.md) for detailed instructions on this step.

### **Step 2: adaptive generation**

We provide ready-to-use scripts for adaptive generation under [`data_generation/generate.sh`](data_generation/generate.sh).  
This script contains both **multi-GPU** and **single-GPU** examples for launching the generation process.

To start adaptive generation, simply run:
```bash
bash data_generation/generate.sh
```

## 🔍 Quality-Driven Contrastive Loss

We use the pretrained CLIP ConvNext-Base model as the encoder to extract feature maps from augmented positive views. These feature maps, with a resolution of 7² from a 224² input, are used to calculate foreground and background attention maps based on PCA.

We randomly sample 10,000 images to compute PCA features. The threshold \( \alpha \) ensures that 40% of the tokens represent the foreground, enabling clear separation.

Use the following command to extract features and compute PCA:

```bash
python tools/clip_pca/extract_features_pca.py \
    --input-list tools/clip_pca/train_sampled_1000cls_10img.txt \
    --num-extract 10000 \
    --patch-size 32 \
    --num-vis 20 \
    --model convnext_base_w \
    --training-data laion2b-s13b-b82k-augreg

### Outputs

- **Extracted Features**: Stored in `features/`.
- **PCA Eigenvectors**: Stored in `eigenvectors/`.
- **Masks, Maps, and Original Images**: Stored in `masks/`, `maps/`, and `original_images/`.

These PCA vectors are used to generate foreground and background attention maps during pretraining. We provide precomputed PCA vectors, which can be found at `tools/clip_pca/pca_results/convnext_base_w_laion2b-s13k-b82k-augreg/eigenvectors/pca_vectors.npy`

## 🔄 Training

Detailed commands for running pretraining and downstream tasks with single or multiple machines/GPUs:

**Training with Multiple GPUs**
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 bash tools/dist_train.sh ${CONFIG_FILE} 8 [PY_ARGS] [--resume /path/to/latest/epoch_{number}.pth]
```

**Training with Multiple Machines**
```shell
CPUS_PER_TASK=8 GPUS_PER_NODE=8 GPUS=16 sh tools/slurm_train.sh $PARTITION $JOBNAME ${CONFIG_FILE} $WORK_DIR [--resume /path/to/latest/epoch_{number}.pth]
```

Ensure to replace `$PARTITION`, `$JOBNAME`, and `$WORK_DIR` with actual values for your setup.
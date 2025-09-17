# calculate noise level

We utilize the pretrained **CLIP ViT-H/14** backbone, which serves as the conditional image encoder in **Stable UnCLIP v2-1**, to determine the proportion of foreground content before image generation. This backbone processes an input resolution of \(224 \times 224\) and generates 256 tokens, each with a dimension of 1280. 

For calculating PCA features needed for foreground-background separation, we randomly sample 10,000 images from the original dataset. The threshold \( \alpha \) in Equation (7) is selected to ensure that foreground tokens account for approximately 40% of the total tokens, providing a clear separation between foreground and background.

### **Step 1: Extract CLIP Image Features and Compute PCA**

We first extract features from 10,000 images using the CLIP ViT-H/14 backbone and then perform PCA analysis.
The calculated PCA vectors act as classifiers for distinguishing between foreground and background regions.

**Command to Extract Features and Perform PCA Analysis:**

```shell
python tools/clip_pca/extract_features_pca.py \
    --input-list data_generation/ada_nl/cc3m_sampled_10000imgs.txt \
    --num-extract 10000 \
    --patch-size 14 \
    --num-vis 20 \
    --model ViT-H-14 \
    --training-data laion2b_s32b_b79k
```

- `--input-list`: Path to the file containing the list of sampled images (`tools/clip_pca/train_sampled_1000cls_10img.txt`).
- `--num-extract 10000`: Specifies the number of images to process.
- `--patch-size 14`: Patch size used by the model.
- `--num-vis 20`: Number of images to visualize.
- `--model ViT-H-14`: Specifies the CLIP model to use.
- `--training-data laion2b_s32b_b79k`: Pretrained weights for the model.

**Outputs:**

- **Extracted Features**: Saved in the `features/` directory.
- **PCA Eigenvectors**: Saved in the `eigenvecters/` directory.
- **Generated Masks, Maps, and Original Images**: Saved in the `masks/`, `maps/`, and `original_images/` directories, respectively.
- **Threshold for Foreground-Background Separation**: During the PCA analysis, a background threshold is also calculated and used for generating masks. This threshold helps to separate foreground from background regions by comparing the PCA-transformed feature values with the threshold. The resulting masks can then be used to compute the foreground ratio for each image in the next steps.

### **Step 2: Determine Suitable Noise Levels for Each Image**

To maintain semantic consistency while ensuring diversity, we determine appropriate noise levels for each image using the PCA vectors and the extracted image features.

#### **2.1 Extract Features for the Entire ImageNet Dataset**

First, we need to extract features for each image in the ImageNet dataset. This process may take around 4 hours with a batch size of 1024, and the extracted features will require approximately 4GB of storage.

**Command to Extract Features:**

```shell
python tools/clip_pca/extract_features_pca.py \
    --input-list data/imagenet/train.txt \
    --num-extract -1 \
    --patch-size 14 \
    --num-vis 20 \
    --model ViT-H-14 \
    --training-data laion2b_s32b_b79k
```

- `--input-list`: Path to the file containing the list of all training images.
- `--num-extract -1`: Processes all images in the list (no limit).
- Other parameters are the same as in Step 1.

#### **2.2 Calculate Foreground Ratios**

Using the previously computed PCA vectors and the foreground-background threshold (`fg_thre`), we calculate the foreground ratio (`fg_ratio`) for each image in the dataset. The `fg_ratio` helps quantify the proportion of foreground content within each image, which will later guide noise level determination for adaptive view generation.

**Command to Calculate `fg_ratio`**:

```shell
python tools/clip_pca/calculate_fgratio.py \
    --input-dir tools/clip_pca/pca_results/ViT-H-14-laion2b_s32b_b79k/ \
    --input-list data/imagenet/train.txt \
    --output-dir data/imagenet/ \
    --fg-thre {computed_threshold}
```
- `--input-dir`: Path to the directory to save extracted features and PCA eigenvecters.
- `--input-list`: Path to the file containing the list of all training images.
- `--output-dir`: Directory where the `fg_ratios.txt` file will be saved.
- `--fg-thre {computed_threshold}`: The foreground-background threshold value (`fg_thre`) calculated from **Step 1** using PCA analysis. This threshold ensures the proper separation of foreground and background regions.

A file named `fg_ratios.txt` will be generated in the specified output directory. This file contains a list of image paths paired with their respective `fg_ratio` values.  
Each line of `fg_ratios.txt` is structured as:  
    ```
    <image_path> <fg_ratio>
    ```
    Example:  
    ```
    data/imagenet/train/img_0001.jpg 0.42
    data/imagenet/train/img_0002.jpg 0.38
    ```
  
#### **2.3 Generate Adaptive Noise Levels**

Finally, we distribute the original `fg_ratios.txt` entries into separate files based on specified ranges and mapping values. Each output file is named after its corresponding mapped noise level value (e.g., `fg_ratios_0.txt`, `fg_ratios_100.txt`, etc.), containing image paths and their `fg_ratio` values that fall into the respective ranges.

**Command to Generate Noise Level Files:**

```shell
python tools/clip_pca/generate_ada_noise_level.py \
    --input-file data/imagenet/fg_ratios.txt \
    --output-dir data/imagenet/
```

- `--input-file`: Path to the `fg_ratios.txt` generated in the previous step.
- `--output-dir`: Directory where the noise level files will be saved.

These files categorize images based on their foreground ratios, allowing us to assign appropriate noise levels during image generation to achieve the desired balance between semantic consistency and diversity.
    ```

# calculate guidance scale




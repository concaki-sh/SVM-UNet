# LFEVM-UNet
LFEVM-UNet: A Local Feature Enhanced Vision Mamba U-Net for Histopathological Gland Segmentation

## Abstract
Colorectal cancer is the third most prevalent malignancy worldwide, and the accuracy of pathological diagnosis plays a critical role in determining clinical treatment strategies. Histopathological images, which provide rich cellular and structural information, are recognized as the gold standard for colorectal cancer grading and diagnosis. Gland morphology serves as a key biomarker for malignancy assessment, making precise gland segmentation a fundamental component of computer-aided pathology. However, existing segmentation methods face persistent challenges due to heterogeneous gland shapes, blurred boundaries, significant size variations, and severe adhesion between glands. To address these issues, we proposed SVM-UNet, a novel segmentation framework based on VM-UNet. The model integrates a Hybrid Atrous Convolution (HAC) module to enhance multi-scale contextual modeling, and incorporates an optimized state space modeling unit, CSS2D, to improve edge and detail representation. Extensive experiments on the CRAG and GlaS datasets have demonstrated that the proposed method outperforms mainstream approaches across multiple metrics, particularly in separating adhesive glands and maintaining boundary continuity. Our code is publicly available at here

## 0. Main Environments
```bash
conda create -n svmunet python=3.8
conda activate svmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
```


## 1. Prepare the dataset

### Glas datasets
- The GLAS dataset from MICCAI 2015 Gland Segmentation Challenge, all the data(include CRAG) can be downloaded from {[google drive](https://drive.google.com/drive/folders/1Q_6MD3tWxjxfAcZ-6qYgvCALnhFpUKIs?usp=sharing)}.
- After downloading the dataset, please place the files into the ./data/Gland/ directory. The expected directory structure and file naming format are shown below (using the GLAS dataset as an example).

- './data/Gland/'
  - train
    - images
      - .bmp
    - labels
      - .bmp
  - test
    - images
      - .bmp
    - labels
      - .bmp
  - test2
    - images
      - .bmp
    - labels
      - .bmp

### CRAG datasets

- For the CRAG dataset, you could follow [MILD-Net: Minimal Information Loss Dilated Network for Gland Instance Segmentation in Colon Histology Images∗]([https://github.com/HuCatoFighting/Swin-Une](https://github.com/XiaoyuZHK/CRAG-Dataset_Aug_ToCOCO)) to download the dataset, or you could download them from {[google drive](https://drive.google.com/drive/folders/1Q_6MD3tWxjxfAcZ-6qYgvCALnhFpUKIs?usp=sharing)}.

- After downloading the datasets, you are supposed to put them into './data/crag/', and the file format reference is as follows.

- './data/crag/'
  - train
    - images
      - .png
    - labels
      - .png
  - test
    - images
      - .png
    - labels
      - .png

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded from [Baidu](https://pan.baidu.com/s/1z9oStFlV0c47dmcRzaRgmQ?pwd=4eid). After that, the pre-trained weights should be stored in './pre_trained_weights/'.
  
## 3. Train the SVM-UNet
```bash
cd SVM-UNet
python train.py  # Train and test VM-UNet on the GLAS or CRAG dataset.
```

**NOTE**: If you want to use the trained checkpoint for inference testing only and save the corresponding test images, you can follow these steps:  

- **In `config_setting`**:  
   - Set the parameter `only_test_and_save_figs` to `True`.  
   - Fill in the path of the trained checkpoint in `best_ckpt_path`.  
   - Specify the save path for test images in `img_save_path`.  

- **Execute the script**:  
   After setting the above parameters, you can run `train.py`.

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'
- The test.py script is used for standalone evaluation of the trained model. It loads the specified model weights and performs inference on the test set without requiring the training process. the best Glas weight can be downloaded from [Baidu](https://pan.baidu.com/s/1srlVWdHTag4vN4RtGZ7Q4A?pwd=8jrf) and CRAG Weight in [Baidu](https://pan.baidu.com/s/1AyuxxAbA7hLX2xbu0A_cQA?pwd=w6i3)

## 5. Acknowledgments

- We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) \ [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)  \ [VM-UNet](https://github.com/JCruan519/VM-UNet/tree/main) or their open-source codes.

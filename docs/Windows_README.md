# CycleGAN and Pix2Pix on Windows: Setup, Training, and Testing Guide

This comprehensive guide outlines the process of setting up, training, and testing CycleGAN and Pix2Pix models on Windows systems.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Model Testing](#model-testing)
6. [Output and Monitoring](#output-and-monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Additional Resources](#additional-resources)

## Prerequisites

- Windows 10 or later
- NVIDIA GPU with CUDA support
- Internet connection
- Administrator privileges

## Environment Setup

1. Install Anaconda:

   - Download from [Anaconda website](https://www.anaconda.com/products/distribution)
   - During installation, select "Add Anaconda to my PATH environment variable"

2. Install NVIDIA GPU Driver:

   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Install and restart your computer

3. Install CUDA Toolkit:

   - Download from [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
   - Add CUDA to your system PATH

4. Create a virtual environment:

   ```
   conda create -n pix2pix python=3.8
   conda activate pix2pix
   ```

5. Install PyTorch:

   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

6. Install dependencies:
   ```
   cd pytorch-CycleGAN-and-pix2pix
   pip install -r requirements.txt
   ```

## Data Preparation

Organize your dataset in `datasets\your_dataset_name`:

- For Pix2Pix: Use `\test`, `\train`, and `\val` folders
- For CycleGAN: Use `TestA`, `TestB`, `TrainA`, `TrainB` folders

## Model Training

Create a file named `train_pix2pix.bat` with the following content:

```batch
@echo off
call C:\path\to\Anaconda3\Scripts\activate.bat pytorch_env

echo Current Python path:
python -c "import sys; print('\n'.join(sys.path))"

echo Checking for torch:
python -c "import torch; print(torch.__version__)"

echo Running training script:
python "C:\path\to\project\pytorch-CycleGAN-and-pix2pix\train.py" ^
--dataroot "C:\path\to\project\pytorch-CycleGAN-and-pix2pix\datasets\your_dataset" ^
--checkpoints_dir "C:\path\to\project\pytorch-CycleGAN-and-pix2pix\checkpoints" ^
--results_dir "C:\path\to\project\results" ^
--name your_experiment_name ^
--model pix2pix ^
--direction AtoB ^
--save_epoch_freq 1 ^
--n_epochs 500 ^
--batch_size 150

pause
```

To run the training script:

1. Open Command Prompt
2. Navigate to the directory containing the .bat file
3. Run the script by typing its name: `train_pix2pix.bat`

## Model Testing

Create a file named `test_pix2pix.bat` with the following content:

```batch
@echo off
call C:\path\to\Anaconda3\Scripts\activate.bat pytorch_env

echo Current Python path:
python -c "import sys; print('\n'.join(sys.path))"

echo Checking for torch:
python -c "import torch; print(torch.__version__)"

echo Running main script:
python "C:\path\to\project\pytorch-CycleGAN-and-pix2pix\test.py" ^
--dataroot "C:\path\to\project\pytorch-CycleGAN-and-pix2pix\datasets\ma-boston" ^
--checkpoints_dir "C:\path\to\project\pytorch-CycleGAN-and-pix2pix\checkpoints" ^
--results_dir "C:\path\to\project\results" ^
--name parcels_pix2pix ^
--model pix2pix ^
--num_test 1000

pause

```

To run the testing script:

1. Open Command Prompt
2. Navigate to the directory containing the .bat file
3. Run the script by typing its name: `[test_pix2pix].bat`

Note: Ensure that you adjust the paths in both scripts to match your specific directory structure.

## Output and Monitoring

1. View training progress: `.\checkpoints\FOLDERNAME\web\index.html`
2. Monitor training logs with `logs-visualised.ipynb` (Work in Progress)
3. Results are saved in the `results` directory

## Troubleshooting

- "conda not found" error: Ensure Anaconda is added to the system PATH
- CUDA-related errors: Verify NVIDIA driver and CUDA versions are compatible with PyTorch
- If CUDA is not detected:
  1. Check GPU driver installation: Run `nvidia-smi` in command prompt
  2. Verify CUDA installation: Run `nvcc --version` in command prompt
  3. Ensure PyTorch CUDA version matches installed CUDA version

## Additional Resources

- [Pix2Pix Documentation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Anaconda Documentation](https://docs.anaconda.com/)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)

For detailed parameter explanations, refer to the `options` directory in the project repository.

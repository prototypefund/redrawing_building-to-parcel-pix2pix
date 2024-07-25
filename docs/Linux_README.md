# CycleGAN and Pix2Pix on Linux: Setup, Training, and Testing Guide

This comprehensive guide outlines the process of setting up, training, and testing CycleGAN and Pix2Pix models on Linux systems, including usage in high-performance computing (HPC) environments.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Model Testing](#model-testing)
6. [Using Batch Job Scripts](#using-batch-job-scripts)
7. [Output and Monitoring](#output-and-monitoring)
8. [GPU Monitoring](#gpu-monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Additional Resources](#additional-resources)

## Prerequisites

- Linux operating system (Ubuntu 18.04 or later recommended)
- NVIDIA GPU with CUDA support
- Internet connection
- Sudo privileges

## Environment Setup

1. Install CUDA and cuDNN:
   - Visit the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
   - Install CUDA and cuDNN following official instructions

2. Install Anaconda:
   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   bash Anaconda3-2023.09-0-Linux-x86_64.sh
   source ~/.bashrc
   ```

3. Create and activate a virtual environment:
   ```bash
   conda create -n pix2pix python=3.8
   conda activate pix2pix
   ```

4. Install PyTorch:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

5. Clone the repository:
   ```bash
   git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
   cd pytorch-CycleGAN-and-pix2pix
   ```

6. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. Organize your dataset in `datasets/your_dataset_name`:
   - For Pix2Pix: Use `/test`, `/train`, and `/val` folders
   - For CycleGAN: Use `TestA`, `TestB`, `TrainA`, `TrainB` folders

2. For image generation, refer to the separate image generation guide.

## Model Training

Navigate to the repository folder:
```bash
cd path/to/pytorch-CycleGAN-and-pix2pix
```

### Pix2Pix Training

```bash
python train.py --dataroot ./datasets/your_dataset \
                --name your_experiment_name \
                --model pix2pix \
                --direction AtoB \
                --save_epoch_freq 1 \
                --n_epochs 500 \
                --batch_size 150
```

### CycleGAN Training

```bash
python train.py --dataroot ./datasets/your_dataset \
                --name your_experiment_name \
                --model cycle_gan \
                --direction AtoB \
                --n_epochs 10 \
                --batch_size 1
```

## Model Testing

```bash
python test.py --dataroot ./datasets/your_test_dataset \
               --name your_experiment_name \
               --model [pix2pix/cycle_gan] \
               --num_test 1000
```

## Using Batch Job Scripts

For HPC environments using SLURM, we provide example scripts:

### Training Script Example (p2p-train-ma-boston-v100.sh)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=train-ma-b-p2p-v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com

module load anaconda3/2022.05 cuda/11.8
source activate /path/to/your/conda/env

python /work/re-blocking/pytorch-CycleGAN-and-pix2pix/train.py \
    --dataroot /work/re-blocking/data/ma-boston \
    --checkpoints_dir /work/re-blocking/checkpoints \
    --name ma-boston-p2p-200-150-v100 \
    --model pix2pix \
    --direction AtoB \
    --save_epoch_freq 1 \
    --continue_train \
    --epoch_count 491 \
    --n_epochs 500 \
    --batch_size 150
```

### Testing Script Example (p2p-test-ma-boston-v100-brooklyn.sh)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:15:00
#SBATCH --job-name=test-ma-b-b-v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com

module load anaconda3/2022.05 cuda/11.8
source activate /path/to/your/conda/env

python /work/re-blocking/pytorch-CycleGAN-and-pix2pix/test.py \
    --dataroot /work/re-blocking/data/ny-brooklyn \
    --checkpoints_dir /work/re-blocking/checkpoints \
    --results_dir /work/re-blocking/results \
    --name ma-boston-p2p-200-150-v100 \
    --model pix2pix \
    --num_test 1000
```

To use these scripts:
1. Save them in your project directory
2. Make them executable: `chmod +x script_name.sh`
3. Submit the job: `sbatch script_name.sh`

## Output and Monitoring

1. View training progress: `./checkpoints/your_experiment_name/web/index.html`
2. Monitor training logs: Use `logs-visualised.ipynb` (Work in Progress)
3. Results are saved in the `results` directory

## GPU Monitoring

Monitor NVIDIA GPU usage:
```bash
watch -n 0.1 nvidia-smi
```

## Troubleshooting

- CUDA errors: Ensure CUDA and PyTorch versions are compatible
- Memory issues: Reduce batch size or image size
- For other issues, consult the official PyTorch and project documentation

## Additional Resources

- [Official CycleGAN and Pix2Pix Repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)

For detailed parameter explanations, refer to the `options` directory in the project repository.


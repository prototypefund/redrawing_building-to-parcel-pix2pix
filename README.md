# building2parcel-pix2pix

This repository contains the implementation of CycleGAN and Pix2Pix models for image-to-image translation tasks. It provides comprehensive setup guides for both Windows and Linux environments.

## Project Overview

This project focuses on implementing and training CycleGAN and Pix2Pix models for various image-to-image translation tasks. These models can be used for a wide range of applications, including style transfer, object transfiguration, season transfer, and photo enhancement.

## Environment Setup

Choose the appropriate guide based on your operating system:

- [Windows Setup Guide](./docs/Windows_README.md)
- [Linux Setup Guide](./docs/Linux_README.md)

## Quick Start

1. Clone this repository:

   ```
   git clone https://github.com/scalable-design-participation-lab/building2parcel-pix2pix.git
   cd re-blocking
   ```

2. Add and initialize the pix2pix submodule:

   ```
   git submodule add https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
   git submodule update --init --recursive
   ```

   This project uses the original [pix2pix repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) as a submodule. Adding it as a submodule allows us to:

   - Keep the original implementation separate from our project-specific code.
   - Easily update to new versions of pix2pix when needed.
   - Clearly distinguish between the original code and our modifications.

3. Follow the setup instructions for your operating system (see links above).

4. Prepare your dataset as described in the setup guides.

## Running the Model

You can run the model either using provided scripts or direct commands.

### Using Scripts

5. Train the model:

   - For Windows, use the `train_pix2pix.bat` script
   - For Linux, use the `train_pix2pix.sh` script

6. Test the model:
   - For Windows, use the `test_pix2pix.bat` script
   - For Linux, use the `test_pix2pix.sh` script

### Using Direct Commands

Alternatively, you can run the model directly using Python commands:

5. Train the model:

   ```
   python train.py --dataroot ./datasets/your_dataset --name your_experiment_name --model pix2pix --direction AtoB
   ```

6. Test the model:
   ```
   python test.py --dataroot ./datasets/your_test_dataset --name your_experiment_name --model pix2pix --direction AtoB
   ```

Replace `your_dataset`, `your_experiment_name`, and other parameters as needed for your specific use case.

Common parameters:

- `--dataroot`: Path to the dataset
- `--name`: Name of the experiment (this will create a folder under `./checkpoints` to store results)
- `--model`: Model to use (pix2pix, cyclegan, etc.)
- `--direction`: AtoB or BtoA

For a full list of available options, refer to the `options` directory in the project or run:

```
python train.py --help
python test.py --help
```

Note: If you're cloning this repository after the submodule has been added, use the following command to clone the repository including all submodules:

```
git clone --recurse-submodules https://github.com/scalable-design-participation-lab/building2parcel-pix2pix.git
```

## Results

After training and testing, you can find the results in the following locations:

- Training progress: `checkpoints/[experiment_name]/web/index.html`
- Test results: `results/[experiment_name]`

## Contributing

We welcome contributions to improve this project. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Original CycleGAN and Pix2Pix implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [PyTorch team](https://pytorch.org/) for their excellent deep learning framework

## Citations

If you use this code for your research, please cite the following papers:

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

For more detailed information on usage, parameters, and advanced features, please refer to the OS-specific README files linked above.

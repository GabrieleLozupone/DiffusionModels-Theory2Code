# 🌊 DiffusionModels-Theory2Code

A comprehensive repository covering both the theoretical foundations and practical implementations of diffusion models in PyTorch. This project bridges mathematical theory with code, guiding you from foundational concepts to advanced techniques in generative AI.

## 📚 Overview

Diffusion models have become one of the most exciting and powerful approaches in generative AI. This repository provides a comprehensive journey from mathematical foundations to practical implementations, featuring rigorous theory alongside executable code. By working through the materials, you'll gain both deep theoretical understanding and hands-on experience implementing various diffusion model techniques on image datasets.

## 🧠 Course Content

The content is organized into a series of modules, each building on the previous ones:

1. **Introduction & Motivation** - Overview of generative models and the motivation behind diffusion models
2. **DDPM (Denoising Diffusion Probabilistic Models)** - Implementation of the foundational diffusion model approach
3. **DDIM (Denoising Diffusion Implicit Models)** - Extension to non-Markovian sampling for faster generation
4. **Classifier Guidance & Conditional Generation** - Methods to guide the generation process
5. **Diffusion Models on CelebA** - Applying diffusion models to human face generation

## ⚙️ Project Structure

```
├── 0X-module.md       # Theory explanations for each module
├── 0X-module.ipynb    # Interactive notebook implementations
├── requirements.txt   # Required Python packages
├── train_celeba_ddpm.py # Training script for CelebA dataset
├── src/               # Source code modules
│   ├── data.py        # Data loading utilities
│   ├── diffusion.py   # Core diffusion process implementations
│   ├── model.py       # Model architectures
│   ├── scheduler.py   # Noise scheduling algorithms
│   ├── training.py    # Training loop utilities
│   └── models/        # Additional model implementations
├── data/              # Dataset storage
│   ├── MNIST/         # MNIST dataset
│   └── celeba/        # CelebA dataset
├── experiments/       # Outputs from training runs
│   └── celeba_64x64/ # Example experiment outputs
└── images/            # Supporting images for documentation
```

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DiffusionModels-Theory2Code
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained models (optional):
   
   All pre-trained models from the `models/` directory can be downloaded from the following Google Drive link:
   
   [Download Pre-trained Models](https://drive.google.com/drive/folders/1BPUNJdd0IoTqa_03ssjvlfDAev9YdmOQ?usp=sharing)
   
   Download and place the models in the `models/` directory to skip training and use them directly.

## 👩‍💻 Usage Guide

### Learning Path

For the best learning experience, follow these steps:

1. Read each `0X-module.md` file to understand the theory
2. Complete the corresponding `0X-module.ipynb` notebook to implement concepts
3. Run experiments with different parameters to build intuition

### Training on CelebA

To train a diffusion model on the CelebA dataset:

```bash
python train_celeba_ddpm.py --image_size 64 --batch_size 64 --epochs 100
```

Check the script for additional command line arguments to customize your training.

### Working with Notebooks

The Jupyter notebooks contain both theoretical explanations and practical implementations. To run them:

```bash
jupyter notebook
```

Then navigate to the desired notebook in the browser interface.

## 🤔 Mathematical Foundation

The workshop covers essential mathematical concepts:

- Forward and reverse diffusion processes
- Evidence Lower Bound (ELBO) optimization
- Variance scheduling techniques
- Non-Markovian sampling strategies
- Classifier guidance for conditional generation

## 📊 Datasets

The workshop uses several datasets:
- **MNIST**: Handwritten digits (basic examples)
- **CelebA**: Human faces (high-resolution generation)

## 🔍 Key Features

- **Theoretical Foundation**: Comprehensive explanation of diffusion model mathematics with complete derivations
- **Theory-to-Code Translation**: Direct mapping between mathematical concepts and their PyTorch implementations
- **Step-by-Step Implementation**: Guided coding exercises with progressive complexity
- **Visualization Tools**: Helper functions to visualize the diffusion process
- **Modular Design**: Code structured for clarity and reusability
- **Scalable Approaches**: From simple to complex datasets

## 📝 References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*.
2. Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. *International Conference on Learning Representations*.
3. Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. *Advances in Neural Information Processing Systems*.

## 🤝 Contributing

Contributions to improve the workshop materials are welcome! Please feel free to submit issues or pull requests.

## 📄 License

[MIT License](LICENSE)

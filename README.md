<!-- # CFSAM-InterBN: Cross-Domain Feature Learning with SAM and InterBN -->
# Sharpness-Aware Minimization for Adversarial Domain Adaptation

A PyTorch implementation of domain adaptation combining SAM (Sharpness-Aware Minimization) optimization and InterBN (Interchangeable Batch Normalization) for robust cross-domain feature learning.

## Features

- **SAM Optimization**: Implementation of Sharpness-Aware Minimization with AMP support
- **InterBN**: Novel batch normalization approach for domain adaptation
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support
- **Advanced Training Features**:
  - Exponential Moving Average (EMA)
  - Cosine learning rate scheduling
  - Sparse regularization
  - Domain adversarial training
- **Experiment Tracking**: Support for TensorBoard and Weights & Biases
- **Development Tools**: Testing, formatting, and linting support

## Project Structure

```
CFSAM_InterBN_Project/
├── configs/                    # Configuration files
│   └── default_config.yaml    # Default training configuration
├── data/                      # Dataset directory
│   └── office31/             # Office-31 dataset
├── datasets/                  # Dataset implementations
│   ├── __init__.py
│   ├── office31.py           # Office-31 dataset loader
│   └── transforms_config.py   # Data augmentation configs
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── feature_extractor.py  # Feature extraction network
│   ├── classifier.py         # Classification head
│   ├── domain_discriminator.py # Domain discriminator
│   └── interbn.py           # InterBN implementation
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── losses.py            # Loss functions
│   ├── sam_optimizer.py     # SAM optimizer
│   └── ema.py              # EMA implementation
├── scripts/                 # Training scripts
│   ├── train.py            # Main training script
│   └── evaluate.py         # Evaluation script
├── logs/                   # Training logs
├── checkpoints/           # Model checkpoints
├── tests/                # Unit tests
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CFSAM-InterBN.git
cd CFSAM-InterBN
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the Office-31 dataset
2. Extract it to `data/office31/`
3. Organize the data as follows:
```
data/office31/
├── amazon/
│   └── images/
├── dslr/
│   └── images/
└── webcam/
    └── images/
```

## Training

1. Basic training:
```bash
python scripts/train.py
```

2. Training with custom config:
```bash
python scripts/train.py --config configs/your_config.yaml
```

### Key Parameters

- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 20)
- `--rho`: SAM rho parameter (default: 0.05)
- `--ema_decay`: EMA decay rate (default: 0.9998)

## Evaluation

To evaluate a trained model:
```bash
python scripts/evaluate.py --checkpoint path/to/checkpoint.pth
```

## Development

- Format code:
```bash
black .
isort .
```

- Run tests:
```bash
pytest tests/
```

## Results

Performance on Office-31 dataset (Accuracy %):

| Method | A → W | W → A | A → D | D → A | W → D | D → W | Avg |
|--------|-------|-------|-------|-------|-------|-------|-----|
| CFSAM-InterBN | 95.2 | 75.6 | 96.1 | 74.8 | 99.3 | 98.7 | 89.9 |



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

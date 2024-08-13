# Towards Strong Certified Defense with Asymmetric Randomization



### Directories

- **archs/**: Contains various neural network architectures used in the project.
- **model_saved/**: Stores saved models, including checkpoints and final models.
- **utils/**: Contains utility functions and scripts that support various tasks within the project.

### Key Python Files

- **architectures.py**: Defines the noise parameter generator architectures used in the experiments, including various neural network designs.
  
  
- **certification_baseline.py**: Implements baseline certification procedures (Cohen et al.) for comparison with advanced methods.
- **certification_certification_noise.py**: Certification with certification-wise noise.
- **certification_dataset_noise.py**: Certification with dataset-wise noise.
- **certification_pattern_noise.py**: Certification with pattern-wise noise.
- **core.py**: Core functionalities for our certification.
- **core_baseline.py**: Core functionalities for Cohen et al.'s certification.
- **datasets.py**: Script to handle dataset loading, preprocessing, and augmentation.
- **noisegenerator.py**: Generates various noise patterns used in the certification process.
- **noises.py**: Defines different types of noises applied during certification.
- **README.md**: The file you are currently reading.
- **train_baselines.py**: Training script for baseline RS models.
- **train_certification_noise.py**: Script for training models with the certification-wise NPG.
- **train_dataset_noise.py**: Script for training models with the dataset-wise NPG.
- **train_pattern_noise.py**: Script for training models with the pattern-wise NPG.



## Running Scripts

#### Train the certification-wise NPG on cifar10 and resnet

```bash
python train_certification_noise.py
cifar10
cifar_resnet110
./model_saved/
--method="PersNoise_isoR"
--model_path=""
--lr=0.01
--batch=100
--sigma=1.0
--epochs=200
--workers=16
--lr_step_size=50
--gpu="0"
--noise_name="Gaussian"
--noisegenerator1_path=""
--noisegenerator2_path=""
--train=1
--IsoMeasure=True
```

#### Certify the Test set with certification-wise NPG

```bash
python certification_certification_noise.py
cifar10
cifar_resnet110
--method="PersNoise_isoR"
--batch=1000
--sigma=1.0
--workers=16
--gpu="0"
--norm=2
--noise_name="Gaussian"
--IsoMeasure=True
```



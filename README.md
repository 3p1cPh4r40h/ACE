# ACE (Automated Chord Extraction)

## Instructions to Run
This code requires pytorch, pandas, scikit-learn, thop and torchinfo libraries. You will also need the **chordino features** and **MIREX labels** files from the [McGill Billboard Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/) which consists of 890 songs.

Place the folders for both the features and labels in the `data` folder in the root project directory. The `process_data.py` file will process the data into a `[data_type]_dataset.pkl` file containing all of the data needed for training; the `clean_data.py` file can be ran afterwards to create the `test_dataloader.pkl`, `train_dataloader.pkl` and `val_dataloader.pkl`.

Note that running `process_data.py` takes a long time; however, `clean_data.py` is much shorter and easier to modify once you have processed the data.

### Running Models
You can run models in several ways:

1. Run a single model:
```bash
python run_models.py --model_type small_dilation
```

2. Run multiple models in sequence:
```bash
python run_models.py --model_type small_dilation carsault multi_dilation
```

3. Run all models in batch mode:
```bash
python run_models.py --batch_mode
```

Additional options:
- `--epochs`: Number of epochs to train (default: 1000)
- `--data_type`: Type of data to use (majmin, majmin7, majmininv, majmin7inv)
- `--loss_hit_epochs`: Number of epochs without improvement before reducing learning rate (default: 50)
- `--early_stop_epochs`: Number of epochs without improvement before early stopping (default: 200)
- `--pretrain_epochs`: Number of epochs for sequence ordering pre-training (default: 1000)

## Instructions to Add New Model
Models are stored in the `model_architecture/architectures` folder. To add a new model:

1. Create a new Python file in the `model_architecture/architectures` folder (e.g., `your_model.py`)
2. Define your model class following the pattern of existing models:
   - Inherit from `nn.Module`
   - Implement `__init__` and `forward` methods
   - Use appropriate PyTorch layers and utilities
   - Include necessary imports (torch, nn, etc.)
   - Add docstrings explaining the model architecture

3. Import your model in `run_models.py` and `test_models.py`:
   ```python
   from model_architecture.architectures.your_model import YourModelClass
   ```

4. Add your model to the `MODEL_TYPES` dictionary in both `run_models.py` and `test_models.py`:
   ```python
   MODEL_TYPES = {
       "your_model": False,  # Set to True if your model requires pre-training
       # ... existing models ...
   }
   ```

5. Add your model to the choices in the argument parsers of both files:
   ```python
   parser.add_argument('--model_type', type=str, default='small_dilation',
                     choices=['small_dilation', 'carsault', 'semi_supervised', 'multi_dilation', 'your_model'],
                     help='Type of model to train (default: small_dilation)')
   ```

6. Add model initialization in the model selection logic of both files:
   ```python
   elif args.model_type == 'your_model':
       model = YourModelClass(num_classes=DATASETS[args.data_type]).to(device)
   ```

7. Your model should accept the following parameters in its constructor:
   - `num_classes`: Number of output classes (chord types)
   - `input_height`: Height of input spectrogram (default=9)
   - `input_width`: Width of input spectrogram (default=24)

8. The model's forward pass should:
   - Accept input tensors of shape (batch_size, height, width)
   - Return logits of shape (batch_size, num_classes)

9. Common utilities available for models:
   - `GaussianNoise`: For data augmentation
   - `BatchNorm2d`: For input normalization
   - `Dropout2d`: For regularization
   - `SqueezeExcitation`: Custom block with sigmoid or softmax attention
   - `MultiDilationBlock`: For applying dilation in parallel

Example model structure:
```python
import torch
import torch.nn as nn
from model_architecture.utils.guassian_noise import GaussianNoise

class YourModel(nn.Module):
    def __init__(self, num_classes, input_height=9, input_width=24):
        super(YourModel, self).__init__()
        # Your model architecture here
        self.fc = nn.Linear(some_size, num_classes)

    def forward(self, x):
        # Your forward pass here
        return self.fc(x)
```

## Idea behind the design
The [Caursault et al](10.3390/electronics10212634) paper, [Chordify app](https://chordify.net/) and a personal desire for the tool along with an interest in convolutional neural networks and signal processing.

The idea is to use short-time fourier transforms (STFT) to process the data, creating a visual representation of pitch intensities over time; hypothetically, there is a correlation between these intensities and all possibly correct chord labels. Convolution has been shown effective at [image classification](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) and [filtering](https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf) tasks, making it an ideal layer choice for extracting features from the STFT medium.

## Ideas to incorporate
- Self-supervised learning
    - Create a base model that learns how to organize chopped up data (try vertical and horizontal chopping).
    - Create a base model that learns how to recognize chord transitions.
- Transfer/reinforcement learning
    - Start with larger context size and use that model to train smaller context size models in addition to labled data valdiation
- Data augmentation
    - Transpose the songs (this is difficult because the data is already processed in the [McGill Billboard Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/)).
- LSTM layer for learning based on context
    - Need to find out how to solve the length issue (songs are variable length)
- NGram Learning

## Tests and results
The [McGill Billboard Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/) has 4 sets of labels it provides, these are the Major/Minor dataset, Major/Minor/Sevenths dataset, Major/Minor/Inversions dataset and the Major/Minor/Sevenths/Inversions dataset; these datasets have 28, 54, 73 and 157 labels respectively, including an X label to represent no chord values (these are stored as `None`, `"N"` and `"X"` in the original dataset and simplified to just `"X"` in the `clean_data.py` file). It should be noted that by default this does not cover every single chord label (only including flat labels, and not including every possible modification), and for some labels there are 2 or less examples in the largest label set. For our purposes we will refer to the datasets as M1, M2, M3 and M4 as seen in the Label Set Table below, working from a the base model provided by Carsault [(a different paper)](https://doi.org/10.48550/arXiv.1911.04973).

### Label Set Table
| Label Set | Included Chord Types | Number of Labels | Labels |
|-----------|----------------------|------------------|--------|
| M-28 | Major, Minor | 28 | N, C:maj, G:maj, F:maj, D:maj, A:maj, E:maj, Bb:maj, Eb:maj, Ab:maj, A:min, B:maj, D:min, E:min, B:min, Db:maj, G:min, C:min, F:min, Gb:maj, Eb:min, Bb:min, Ab:min, Cb:maj, Db:min, Fb:maj, Gb:min, Cb:min |
| M-54 | Major, Minor, 7th | 54 | N, C:maj, F:maj, G:maj, D:maj, A:maj, E:maj, Bb:maj, Ab:maj, Eb:maj, A:min, B:maj, Db:maj, D:min, E:min, B:min, C:maj7, G:maj7, D:maj7, F:maj7, A:maj7, E:maj7, Eb:maj7, Ab:maj7, D:min7, E:min7, A:min7, Gb:maj, G:min, Bb:maj7, B:maj7, G:min7, C:min, B:min7, Eb:min7, F:min, C:min7, Db:maj7, F:min7, Bb:min7, Eb:min, Bb:min, Gb:maj7, Ab:min7, Ab:min, Cb:maj, Db:min7, Cb:maj7, Fb:maj, Db:min, Gb:min, Cb:min, Gb:min7, Fb:maj7 |
| M-73 | Major, Minor, Inversions | 73 | N, C:maj, G:maj, F:maj, D:maj, A:maj, E:maj, Bb:maj, Eb:maj, Ab:maj, A:min, B:maj, E:min, D:min, B:min, Db:maj, G:min, C:min, Gb:maj, F:min, Eb:min, Bb:min, D:maj/5, Ab:min, F:maj/3, F:maj/5, D:min/5, C:maj/3, Bb:maj/5, D:maj/3, E:maj/3, C:maj/5, G:maj/3, Db:maj/5, G:maj/5, Cb:maj, A:maj/5, A:maj/3, Db:min, E:maj/5, F:min/5, Ab:maj/3, Eb:maj/5, Ab:maj/5, D:min/b3, Bb:maj/3, B:maj/5, Fb:maj, E:min/b3, Gb:maj/5, Eb:maj/3, E:min/5, A:min/b3, Db:maj/3, Gb:min, G:min/5, B:min/5, C:min/b3, Eb:min/5, B:min/b3, B:maj/3, A:min/5, G:min/b3, F:min/b3, Ab:min/5, C:min/5, Gb:maj/3, Eb:min/b3, Cb:maj/5, Ab:min/b3, Db:min/5, Bb:min/5, Cb:min |
| M-157 | Major, Minor, 7th, Inversions | 157| N, C:maj, G:maj, F:maj, D:maj, A:maj, E:maj, Bb:maj, Eb:maj, Ab:maj, A:min, B:maj, B:min, E:min, Db:maj, D:min, G:maj7, C:maj7, D:maj7, F:maj7, A:maj7, E:maj7, Eb:maj7, Ab:maj7, E:min7, D:min7, A:min7, Gb:maj, G:min, G:min7, B:maj7, Bb:maj7, C:min, Eb:min7, B:min7, C:min7, F:min, Db:maj7, F:min7, Bb:min7, Eb:min, Bb:min, Gb:maj7, F:maj/5, F:maj/3, Ab:min7, D:min/5, C:maj/5, Db:maj/5, D:maj/3, E:maj/3, Bb:maj/5, G:maj/3, Ab:min, G:maj/5, C:maj/3, A:maj/5, A:maj/3, E:maj/5, Cb:maj, Db:min7, Ab:maj/3, Ab:maj/5, F:min7/5, Eb:maj/5, D:min/b3, C:maj7/3, Bb:maj/3, Bb:maj7/7, Cb:maj7, Fb:maj, F:min/5, D:min7/5, Bb:maj7/5, E:min/b3, B:maj/5, Gb:maj/5, Db:min, Gb:min, A:min/b3, D:min7/b7, D:maj7/3, F:maj7/3, Eb:maj/3, E:min7/3, G:maj7/3, G:min/5, B:min/5, Db:maj/3, E:maj7/b7, B:min7/b7, G:maj7/3, F:maj7/5, D:maj7/5, Eb:maj7/5, Db:maj7/3, A:maj7/7, A:maj7/5, B:maj7/5, C:maj7/b7, A:maj7/3, D:maj7/b7, A:min7/b7, G:min/b3, E:min7/5, D:min7/b3, B:maj7/b7, G:maj7/b7, Eb:maj7/3, B:min7/b3, G:min7/5, B:maj7/3, A:maj7/b7, Ab:maj7/b7, C:min/5, Gb:maj/3, B:min/b3, Ab:min/5, Gb:maj7/5, B:maj/3, Db:maj7/7, Eb:maj7/b7, C:maj7/5, Bb:maj7/3, A:min7/5, Eb:min/b3, B:min7/5, A:min/5, F:min/b3, Ab:maj7/5, Ab:maj7/3, Db:maj7/b7, Cb:maj/5, E:min7/b3, A:min7/b3, Ab:min/b3, F:min7/b3, C:min7/b3, Eb:min7/b7, Db:min/5, E:maj7/5, Ab:maj7/7, Bb:min7/b7, Bb:min/5, F:maj7/b7, Cb:min, Gb:min7, Fb:maj7, F:maj7/7, Db:maj7/5, Ab:min7/5, G:min7/b3, G:min7/b7 |

### Label Distribution Plots
The following plots show the distribution of class labels for each dataset:

- **M-28 (majmin)**: ![Label Distribution for M1](data/majmin/label_distribution_majmin.png)
- **M-54 (majmin7)**: ![Label Distribution for M2](data/majmin7/label_distribution_majmin7.png)
- **M-73 (majmininv)**: ![Label Distribution for M3](data/majmininv/label_distribution_majmininv.png)
- **M-157 (majmin7inv)**: ![Label Distribution for M4](data/majmin7inv/label_distribution_majmin7inv.png)

We stratify the test/validation and training datasets to keep the class representation consistent throughout the datasets.

### Model Comparison Table

#### M-28 Dataset
| Model (M-28)                    | Accuracy | F1 Score | GFLOPs | Parameters |
|--------------------------------|----------|----------|--------|------------|
| [Carsault](10.3390/electronics10212634) | 0.6731   | 0.5879   | 0.0019 | 263,122    |
| Small Dilation                 | 0.6874   | 0.6103   | 0.0065 | 1,009,618  |
| Small Dilation First*          | 0.6347   | 0.4959   | 0.0065 | 1,009,618  |
| Small Dilation Second*         | 0.6333   | 0.4971   | 0.0065 | 1,009,618  |
| Small Dilation Last*           | 0.6360   | 0.4968   | 0.0065 | 1,009,618  |
| Small Dilation First Two*      | 0.6341   | 0.4935   | 0.0065 | 1,009,618  |
| Small Dilation Last Two*       | 0.6332   | 0.4921   | 0.0065 | 1,009,618  |
| Small Dilation First Last*     | 0.6335   | 0.4935   | 0.0065 | 1,009,618  |
| Multi Dilation                 | 0.6475   | 0.5631   | 0.0072 | 1,011,202  |
| Multi Dilation 248             | 0.6575   | 0.5202   | 0.0072 | 1,011,202  |
| Multi Dilation 2832            | 0.6567   | 0.5164   | 0.0072 | 1,011,202  |
| Multi Dilation 4816            | 0.6561   | 0.5135   | 0.0072 | 1,011,202  |
| Multi Dilation 81632           | 0.6553   | 0.5151   | 0.0072 | 1,011,202  |
| Semi Supervised                | 0.6131   | 0.4724   | 0.0019 | 290,986    |
| Early Squeeze                  | 0.6492   | 0.5680   | 0.0072 | 1,011,202  |
| Mid Squeeze                    | 0.6498   | 0.5684   | 0.0072 | 1,011,346  |
| Late Squeeze                   | 0.6441   | 0.5542   | 0.0072 | 1,011,346  |
| Early Squeeze Softmax*         | 0.6341   | 0.4847   | 0.0072 | 1,011,202  |
| Mid Squeeze Softmax*           | 0.4917   | 0.2711   | 0.0072 | 1,011,202  |
| Late Squeeze Softmax*          | 0.5644   | 0.2709   | 0.0072 | 1,011,202  |
| Multi Dilation Early Squeeze Softmax* | 0.6341 | 0.4847 | 0.0072 | 1,011,202  |
| Multi Dilation Early Squeeze Sigmoid* | 0.6236 | 0.4767 | 0.0072 | 1,011,202  |
| Multi Dilation Mid Squeeze Softmax* | 0.4917 | 0.2711 | 0.0072 | 1,011,202  |
| Multi Dilation Mid Squeeze Sigmoid* | 0.6262 | 0.4792 | 0.0072 | 1,011,202  |
| Multi Dilation Late Squeeze Softmax* | 0.5644 | 0.2709 | 0.0072 | 1,011,202  |
| Multi Dilation Late Squeeze Sigmoid* | 0.1952 | 0.0874 | 0.0072 | 1,011,202  |

*Models marked with an asterisk were only run for 10 epochs.

*See the model_stats_majmin.txt file in each ModelResults/<model_name>/majmin/ directory for full details.*

### Loss Graphs

#### M-28 Dataset
| Model (M-28)                     | Loss Graph |
|---------------------------|------------|
| Carsault | ![Carsault - Majmin Loss](ModelResults/carsault/majmin/loss_plot_majmin_classification.png) |
| Small Dilation | ![Small Dilation - M-28](ModelResults/small_dilation/majmin/loss_plot_majmin_classification.png) |
| Small Dilation First | ![Small Dilation First - M-28](ModelResults/small_dilation_first/majmin/loss_plot_majmin_classification.png) |
| Small Dilation Second | ![Small Dilation Second - M-28](ModelResults/small_dilation_second/majmin/loss_plot_majmin_classification.png) |
| Small Dilation Last | ![Small Dilation Last - M-28](ModelResults/small_dilation_last/majmin/loss_plot_majmin_classification.png) |
| Small Dilation First Two | ![Small Dilation First Two - M-28](ModelResults/small_dilation_first_two/majmin/loss_plot_majmin_classification.png) |
| Small Dilation Last Two | ![Small Dilation Last Two - M-28](ModelResults/small_dilation_last_two/majmin/loss_plot_majmin_classification.png) |
| Small Dilation First Last | ![Small Dilation First Last - M-28](ModelResults/small_dilation_first_last/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation | ![Multi Dilation - M-28](ModelResults/multi_dilation/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation 248 | ![Multi Dilation 248 - M-28](ModelResults/multi_dilation_248/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation 2832 | ![Multi Dilation 2832 - M-28](ModelResults/multi_dilation_2832/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation 4816 | ![Multi Dilation 4816 - M-28](ModelResults/multi_dilation_4816/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation 81632 | ![Multi Dilation 81632 - M-28](ModelResults/multi_dilation_81632/majmin/loss_plot_majmin_classification.png) |
| Semi Supervised (sequence) | ![Semi Supervised (sequence) - M-28](ModelResults/semi_supervised/majmin/loss_plot_majmin_sequence.png) |
| Semi Supervised | ![Semi Supervised - M-28](ModelResults/semi_supervised/majmin/loss_plot_majmin_classification.png) |
| Early Squeeze | ![Early Squeeze - M-28](ModelResults/early_squeeze/majmin/loss_plot_majmin_classification.png) |
| Mid Squeeze | ![Mid Squeeze - M-28](ModelResults/mid_squeeze/majmin/loss_plot_majmin_classification.png) |
| Late Squeeze | ![Late Squeeze - M-28](ModelResults/late_squeeze/majmin/loss_plot_majmin_classification.png) |
| Early Squeeze Softmax | ![Early Squeeze Softmax - M-28](ModelResults/early_squeeze_softmax/majmin/loss_plot_majmin_classification.png) |
| Mid Squeeze Softmax | ![Mid Squeeze Softmax - M-28](ModelResults/mid_squeeze_softmax/majmin/loss_plot_majmin_classification.png) |
| Late Squeeze Softmax | ![Late Squeeze Softmax - M-28](ModelResults/late_squeeze_softmax/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation Early Squeeze Softmax | ![Multi Dilation Early Squeeze Softmax - M-28](ModelResults/multi_dilation_early_squeeze_softmax/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation Early Squeeze Sigmoid | ![Multi Dilation Early Squeeze Sigmoid - M-28](ModelResults/multi_dilation_early_squeeze_sigmoid/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation Mid Squeeze Softmax | ![Multi Dilation Mid Squeeze Softmax - M-28](ModelResults/multi_dilation_mid_squeeze_softmax/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation Mid Squeeze Sigmoid | ![Multi Dilation Mid Squeeze Sigmoid - M-28](ModelResults/multi_dilation_mid_squeeze_sigmoid/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation Late Squeeze Softmax | ![Multi Dilation Late Squeeze Softmax - M-28](ModelResults/multi_dilation_late_squeeze_softmax/majmin/loss_plot_majmin_classification.png) |
| Multi Dilation Late Squeeze Sigmoid | ![Multi Dilation Late Squeeze Sigmoid - M-28](ModelResults/multi_dilation_late_squeeze_sigmoid/majmin/loss_plot_majmin_classification.png) |

### Test Results

#### M-28 Dataset
Below are sample test results showing input spectrograms and prediction probabilities for each model:

| Model (M-28) | Sample Results |
|--------------|----------------|
| Carsault | ![Carsault Sample 1](ModelTestResults/carsault/majmin/final_model/sample_1_input.png) ![Carsault Sample 1 Probabilities](ModelTestResults/carsault/majmin/final_model/sample_1_probabilities.png) |
| Small Dilation | ![Small Dilation Sample 1](ModelTestResults/small_dilation/majmin/final_model/sample_1_input.png) ![Small Dilation Sample 1 Probabilities](ModelTestResults/small_dilation/majmin/final_model/sample_1_probabilities.png) |
| Small Dilation First* | ![Small Dilation First Sample 1](ModelTestResults/small_dilation_first/majmin/final_model/sample_1_input.png) ![Small Dilation First Sample 1 Probabilities](ModelTestResults/small_dilation_first/majmin/final_model/sample_1_probabilities.png) |
| Small Dilation Second* | ![Small Dilation Second Sample 1](ModelTestResults/small_dilation_second/majmin/final_model/sample_1_input.png) ![Small Dilation Second Sample 1 Probabilities](ModelTestResults/small_dilation_second/majmin/final_model/sample_1_probabilities.png) |
| Small Dilation Last* | ![Small Dilation Last Sample 1](ModelTestResults/small_dilation_last/majmin/final_model/sample_1_input.png) ![Small Dilation Last Sample 1 Probabilities](ModelTestResults/small_dilation_last/majmin/final_model/sample_1_probabilities.png) |
| Small Dilation First Two* | ![Small Dilation First Two Sample 1](ModelTestResults/small_dilation_first_two/majmin/final_model/sample_1_input.png) ![Small Dilation First Two Sample 1 Probabilities](ModelTestResults/small_dilation_first_two/majmin/final_model/sample_1_probabilities.png) |
| Small Dilation Last Two* | ![Small Dilation Last Two Sample 1](ModelTestResults/small_dilation_last_two/majmin/final_model/sample_1_input.png) ![Small Dilation Last Two Sample 1 Probabilities](ModelTestResults/small_dilation_last_two/majmin/final_model/sample_1_probabilities.png) |
| Small Dilation First Last* | ![Small Dilation First Last Sample 1](ModelTestResults/small_dilation_first_last/majmin/final_model/sample_1_input.png) ![Small Dilation First Last Sample 1 Probabilities](ModelTestResults/small_dilation_first_last/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation | ![Multi Dilation Sample 1](ModelTestResults/multi_dilation/majmin/final_model/sample_1_input.png) ![Multi Dilation Sample 1 Probabilities](ModelTestResults/multi_dilation/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation 248 | ![Multi Dilation 248 Sample 1](ModelTestResults/multi_dilation_248/majmin/final_model/sample_1_input.png) ![Multi Dilation 248 Sample 1 Probabilities](ModelTestResults/multi_dilation_248/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation 2832 | ![Multi Dilation 2832 Sample 1](ModelTestResults/multi_dilation_2832/majmin/final_model/sample_1_input.png) ![Multi Dilation 2832 Sample 1 Probabilities](ModelTestResults/multi_dilation_2832/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation 4816 | ![Multi Dilation 4816 Sample 1](ModelTestResults/multi_dilation_4816/majmin/final_model/sample_1_input.png) ![Multi Dilation 4816 Sample 1 Probabilities](ModelTestResults/multi_dilation_4816/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation 81632 | ![Multi Dilation 81632 Sample 1](ModelTestResults/multi_dilation_81632/majmin/final_model/sample_1_input.png) ![Multi Dilation 81632 Sample 1 Probabilities](ModelTestResults/multi_dilation_81632/majmin/final_model/sample_1_probabilities.png) |
| Semi Supervised | ![Semi Supervised Sample 1](ModelTestResults/semi_supervised/majmin/final_model/sample_1_input.png) ![Semi Supervised Sample 1 Probabilities](ModelTestResults/semi_supervised/majmin/final_model/sample_1_probabilities.png) |
| Early Squeeze | ![Early Squeeze Sample 1](ModelTestResults/early_squeeze/majmin/final_model/sample_1_input.png) ![Early Squeeze Sample 1 Probabilities](ModelTestResults/early_squeeze/majmin/final_model/sample_1_probabilities.png) |
| Mid Squeeze | ![Mid Squeeze Sample 1](ModelTestResults/mid_squeeze/majmin/final_model/sample_1_input.png) ![Mid Squeeze Sample 1 Probabilities](ModelTestResults/mid_squeeze/majmin/final_model/sample_1_probabilities.png) |
| Late Squeeze | ![Late Squeeze Sample 1](ModelTestResults/late_squeeze/majmin/final_model/sample_1_input.png) ![Late Squeeze Sample 1 Probabilities](ModelTestResults/late_squeeze/majmin/final_model/sample_1_probabilities.png) |
| Early Squeeze Softmax* | ![Early Squeeze Softmax Sample 1](ModelTestResults/early_squeeze_softmax/majmin/final_model/sample_1_input.png) ![Early Squeeze Softmax Sample 1 Probabilities](ModelTestResults/early_squeeze_softmax/majmin/final_model/sample_1_probabilities.png) |
| Mid Squeeze Softmax* | ![Mid Squeeze Softmax Sample 1](ModelTestResults/mid_squeeze_softmax/majmin/final_model/sample_1_input.png) ![Mid Squeeze Softmax Sample 1 Probabilities](ModelTestResults/mid_squeeze_softmax/majmin/final_model/sample_1_probabilities.png) |
| Late Squeeze Softmax* | ![Late Squeeze Softmax Sample 1](ModelTestResults/late_squeeze_softmax/majmin/final_model/sample_1_input.png) ![Late Squeeze Softmax Sample 1 Probabilities](ModelTestResults/late_squeeze_softmax/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation Early Squeeze Softmax* | ![Multi Dilation Early Squeeze Softmax Sample 1](ModelTestResults/multi_dilation_early_squeeze_softmax/majmin/final_model/sample_1_input.png) ![Multi Dilation Early Squeeze Softmax Sample 1 Probabilities](ModelTestResults/multi_dilation_early_squeeze_softmax/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation Early Squeeze Sigmoid* | ![Multi Dilation Early Squeeze Sigmoid Sample 1](ModelTestResults/multi_dilation_early_squeeze_sigmoid/majmin/final_model/sample_1_input.png) ![Multi Dilation Early Squeeze Sigmoid Sample 1 Probabilities](ModelTestResults/multi_dilation_early_squeeze_sigmoid/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation Mid Squeeze Softmax* | ![Multi Dilation Mid Squeeze Softmax Sample 1](ModelTestResults/multi_dilation_mid_squeeze_softmax/majmin/final_model/sample_1_input.png) ![Multi Dilation Mid Squeeze Softmax Sample 1 Probabilities](ModelTestResults/multi_dilation_mid_squeeze_softmax/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation Mid Squeeze Sigmoid* | ![Multi Dilation Mid Squeeze Sigmoid Sample 1](ModelTestResults/multi_dilation_mid_squeeze_sigmoid/majmin/final_model/sample_1_input.png) ![Multi Dilation Mid Squeeze Sigmoid Sample 1 Probabilities](ModelTestResults/multi_dilation_mid_squeeze_sigmoid/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation Late Squeeze Softmax* | ![Multi Dilation Late Squeeze Softmax Sample 1](ModelTestResults/multi_dilation_late_squeeze_softmax/majmin/final_model/sample_1_input.png) ![Multi Dilation Late Squeeze Softmax Sample 1 Probabilities](ModelTestResults/multi_dilation_late_squeeze_softmax/majmin/final_model/sample_1_probabilities.png) |
| Multi Dilation Late Squeeze Sigmoid* | ![Multi Dilation Late Squeeze Sigmoid Sample 1](ModelTestResults/multi_dilation_late_squeeze_sigmoid/majmin/final_model/sample_1_input.png) ![Multi Dilation Late Squeeze Sigmoid Sample 1 Probabilities](ModelTestResults/multi_dilation_late_squeeze_sigmoid/majmin/final_model/sample_1_probabilities.png) |

*Note: Each sample shows the input spectrogram (left) and the model's predicted probability distribution across all 28 chord classes (right). The true chord label and predicted chord label are shown in the titles. Models marked with an asterisk were only run for 10 epochs.

#### Pretraining Results (M-28 Dataset)
The semi-supervised model uses a pretraining phase to learn chord sequence patterns. Below are sample results showing the pretraining process:

| Model (M-28) | Pretraining Results |
|--------------|---------------------|
| Semi Supervised | ![Original Spectrogram](ModelTestResults/semi_supervised/majmin/pretrained_model/sample_1_original.png) ![Shuffled Spectrogram](ModelTestResults/semi_supervised/majmin/pretrained_model/sample_1_shuffled.png) ![Reconstructed Spectrogram](ModelTestResults/semi_supervised/majmin/pretrained_model/sample_1_reconstructed.png) |


## Immediate Goals
-  Find a train/test/val split based on literature
- CNN-LSTM from the [Caursault et al 2024](10.3390/electronics10212634) paper.
- Show 10 examples for each class randomly pulled from training and testing sets (and visualize how they were generated).
- Try using concurrent dilation layers with different rates (as opposed to a single rate and feeding dilation layers into each other). Can feed into your attention mechanism.
-  Add multi dilation w/ attention (early, mid, late)
- Add conv block after multi-dilation (experiment with batch-norm position)
- Create a program to transpose the data (working from the central 12 bins, moving up 6 semitones and down 6 semitones while adjusting labels accordingly).
    - Could also add logic to create copies of every piece of data along with all of their labels (i.e. E=Fb, so any instances of the label E could be copied with the label Fb and still be valid); however, this has other possible side effects, as future labels depend on previous labels when transcribing chords.
-  CQT explanation ([Link](https://www.ee.columbia.edu/~dpwe/papers/Brown91-cqt.pdf))
-  Comparison to MIREX standard
-  Show samples and prediction on samples (for both the pre-training and regular training steps)
- Experiment with softmax and sigmoid attention (before or after combining branches?)

## References

[1]
S. Bhardwaj, S. M. Salim, D. T. A. Khan, and S. J. Masoudian, 'Automated Music Generation using Deep Learning', 2022, pp. 193–198.

[2]
J. Choi and K. Lee, 'Pop2Piano : Pop Audio-Based Piano Cover Generation', 2023, pp. 1–5.

[3]
Y. Wu, T. Carsault, and K. Yoshii, 'Automatic Chord Estimation Based on a Frame-wise Convolutional Recurrent Neural Network with Non-Aligned Annotations', 2019, pp. 1–5.

[4]
H. Yamaga, T. Momma, K. Kojima, and Y. Itoh, 'Ensemble of Transformer and Convolutional Recurrent Neural Network for Improving Discrimination Accuracy in Automatic Chord Recognition', 2023, pp. 2299–2305.

[5]
E. J. Humphrey and J. P. Bello, 'Rethinking Automatic Chord Recognition with Convolutional Neural Networks', 2012, vol. 2, pp. 357–362.

[6]
S. Maruo, K. Yoshii, K. Itoyama, M. Mauch, and M. Goto, 'A feedback framework for improved chord recognition based on NMF-based approximate note transcription', 2015, pp. 196–200.

[7]
G. Durán and P. de la Cuadra, 'Transcribing Lead Sheet-Like Chord Progressions of Jazz Recordings', Computer Music Journal, vol. 44, pp. 26–42, 2021.

[8]
K. Vaca, A. Gajjar, and X. Yang, 'Real-Time Automatic Music Transcription (AMT) with Zync FPGA', 2019, pp. 378–384.

[9]
K. Shibata et al., 'Joint Transcription of Lead, Bass, and Rhythm Guitars Based on a Factorial Hidden Semi-Markov Model', 2019, pp. 236–240.

[10]
H. Pedroza, W. Abreu, R. M. Corey, and I. R. Roman, 'Guitar-TECHS: An Electric Guitar Dataset Covering Techniques, Musical Excerpts, Chords and Scales Using a Diverse Array of Hardware', 2025, pp. 1–5.

[11]
J. Sun, H. Li, and L. Lei, 'Key detection through pitch class distribution model and ANN', 2009, pp. 1–6.

[12]
E. J. Humphrey and J. P. Bello, 'From music audio to chord tablature: Teaching deep convolutional networks toplay guitar', in 2014 IEEE International Conference on Acoustic, Speech and Signal Processing (ICASSP), 2014.

[13]
M. Bortolozzo, R. Schramm, and C. R. Jung, 'Improving the Classification of Rare Chords With Unlabeled Data', in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) |, 2021.

[14]
T. Carsault, J. Nika, P. Esling, and G. Assayag, 'Combining Real-Time Extraction and Prediction of Musical Chord Progressions for Creative Applications', 2021, vol. 10.

[15]
Y. Jadhav, A. Patel, R. H. Jhaveri, and R. Raut, 'Transfer Learning for Audio Waveform to Guitar Chord Spectrograms Using the Convolution Neural Network', Mobile Information Systems, vol. 2022, pp. 1–11, Aug. 2022.

[16]
Y.-S. Lee, Y.-L. Chiang, P.-R. Lin, C.-H. Lin, and T.-C. Tai, 'Robust and efficient content-based music retrieval system', APSIPA Transactions on Signal and Information Processing, vol. 5, no. 1, 2016.

[17]
N. Li, 'Generative Adversarial Network for Musical Notation Recognition during Music Teaching', Computational Intelligence and Neuroscience, vol. 2022, pp. 1–9, Jun. 2022.

[18]
H. Mukherjee et al., 'Music chord inversion shape identification with LSTM-RNN', Procedia Computer Science, vol. 167, pp. 607–615, 2020.

[19]
R. Nishikimi, E. Nakamura, M. Goto, and K. Yoshii, 'Audio-to-score singing transcription based on a CRNN-HSMM hybrid model', APSIPA Transactions on Signal and Information Processing, vol. 10, no. 1, 2021.

[20]
Y. Ojima, E. Nakamura, K. Itoyama, and K. Yoshii, 'Chord-aware automatic music transcription based on hierarchical Bayesian integration of acoustic and language models', APSIPA Transactions on Signal and Information Processing, vol. 7, no. 1, 2018.

[21]
J. Pauwels, J.-P. Martens, and M. Leman, 'The Influence of Chord Duration Modeling on Chord and Local Key Extraction', in 2011 10th International Conference on Machine Learning and Applications and Workshops, 2011, pp. 136–141.

[22]
A. Perez, H. L. Ma, S. Zawaduk, and M. R. W. Dawson, 'How Do Artificial Neural Networks Classify Musical Triads? A Case Study in Eluding Bonini's Paradox', Cognitive Science, vol. 47, no. 1, Jan. 2023.

[23]
J. de Berardinis, A. Meroño-Peñuela, A. Poltronieri, and V. Presutti, 'ChoCo: a Chord Corpus and a Data Transformation Workflow for Musical Harmony Knowledge Graphs', Scientific Data, vol. 10, no. 1, Sep. 2023.

[24]
I. Rida, R. Herault, and G. Gasso, 'Supervised Music Chord Recognition', in 2014 13th International Conference on Machine Learning and Applications, 2014, pp. 336–341.

[25]
S. Shi, S. Xi, and S.-B. Tsai, 'Research on Autoarrangement System of Accompaniment Chords Based on Hidden Markov Model with Machine Learning', Mathematical Problems in Engineering, vol. 2021, pp. 1–10, Oct. 2021.

[26]
W. Wang, 'Music chord sequence recognition method based on audio feature extraction algorithm', in 2023 IEEE International Conference on Control, Electronics and Computer Technology (ICCECT), 2023.

[27]
Y. Wu and K. Yoshii, 'Joint Chord and Key Estimation Based on a Hierarchical Variational Autoencoder with Multi-task Learning', APSIPA Transactions on Signal and Information Processing, vol. 11, no. 1, 2022.

[28]
Y. Yu, R. Zimmermann, Y. Wang, and V. Oria, 'Recognition and Summarization of Chord Progressions and Their Application to Music Information Retrieval', in 2012 IEEE International Symposium on Multimedia, 2012, pp. 9–16.

[29]
C. Zhuang, 'GCA:A chord music generation algorithm based on double-layer LSTM', in 2021 3rd International Conference on Advances in Computer Technology, Information Science and Communication (CTISC), 2021.

[30]
G. Brunner, Y. Wang, R. Wattenhofer, and J. Wiesendanger, 'JamBot: Music Theory Aware Chord Based Generation of Polyphonic Music with LSTMs', in 2017 IEEE 29th International Conference on Tools with Artificial Intelligence (ICTAI), 2017, pp. 519–526.

[31]
T. Gagnon, S. Larouche, and R. Lefebvre, 'A neural network approach for preclassification in musical chords recognition', 2003, vol. 2, pp. 2106-2109 Vol.2.

[32]
F. Korzeniowski, D. R. W. Sears, and G. Widmer, 'A Large-Scale Study of Language Models for Chord Prediction', CoRR, vol. abs/1804.01849, 2018.

[33]
Y. Wu, E. Nakamura, and K. Yoshii, 'A Variational Autoencoder for Joint Chord and Key Estimation from Audio Chromagrams', 2020, pp. 500–506.

[34]
V. Eremenko, E. Demirel, B. Bozkurt, and X. Serra, 'Audio-Aligned Jazz Harmony Dataset for Automatic Chord Transcription and Corpus-based Research', in Proceedings of the 19th International Society for Music Information Retrieval Conference, ISMIR 2018, Paris, France, September 23-27, 2018, 2018, pp. 483–490.

[35]
N. Orio, 'Music Retrieval: A Tutorial and Review', Found. Trends Inf. Retr., vol. 1, no. 1, pp. 1–90, 2006.

[36]
Z. Wang et al., 'POP909: A Pop-Song Dataset for Music Arrangement Generation', in Proceedings of the 21th International Society for Music Information Retrieval Conference, ISMIR 2020, Montreal, Canada, October 11-16, 2020, 2020, pp. 38–45.

[37]
L. Oudre, C. Fevotte, and Y. Grenier, 'Probabilistic Template-Based Chord Recognition', IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, pp. 2249–2259, Nov. 2011.

[38]
G. R. M., K. S. Rao, and P. P. Das, 'Harmonic-Percussive Source Separation of Polyphonic Music by Suppressing Impulsive Noise Events', in 19th Annual Conference of the International Speech Communication Association, Interspeech 2018, Hyderabad, India, September 2-6, 2018, 2018, pp. 831–835.

[39]
J. Sleep, 'Automatic Music Transcription with Convolutional Neural Networks using Intuitive Filter Shapes', Robert E. Kennedy Library, Cal Poly, 2017.

[40]
J. C. Brown, 'Calculation of a constant Q spectral transform', The Journal of the Acoustical Society of America, vol. 89, no. 1, pp. 425–434, Jan. 1991.

[41]
Y. Wu, T. Carsault, E. Nakamura, and K. Yoshii, 'Semi-Supervised Neural Chord Estimation Based on a Variational Autoencoder With Latent Chord Labels and Features', IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2956–2966, 2020.

[42]
Y. Yu, R. Zimmermann, Y. Wang, and V. Oria, 'Scalable Content-Based Music Retrieval Using Chord Progression Histogram and Tree-Structure LSH', IEEE Transactions on Multimedia, vol. 15, pp. 1969–1981, Dec. 2013.

[43]
L. C. Reghunath and R. Rajan, 'Predominant audio source separation in polyphonic music', EURASIP Journal on Audio, Speech, and Music Processing, vol. 2023, no. 1, Nov. 2023.

[44]
E. Tverdokhleb, N. Myronova, and T. Fedoronchak, 'Music signal processing to obtain its chorded representation', 2017, pp. 301–304.

[45]
J.-Q. Deng and Y.-K. Kwok, 'LARGE VOCABULARY AUTOMATIC CHORD ESTIMATION WITH AN EVEN CHANCE TRAINING SCHEME', in Proceedings of the 18th International Society for Music Information Retrieval Conference, ISMIR 2017, Suzhou, China, October 23-27, 2017, 2017, pp. 531–536.

[46]
H. V. Koops, W. B. de Haas, J. Bransen, and A. Volk, 'Automatic chord label personalization through deep learning of shared harmonic interval profiles', Neural Comput. Appl., vol. 32, no. 4, pp. 929–939, 2020.

[47]
E. Row, J. Tang, and G. Fazekas, 'JAZZVAR: A Dataset of Variations found within Solo Piano Performances of Jazz Standards for Music Overpainting', CoRR, vol. abs/2307.09670, 2023.

[48]
G. Reis, N. Fonseca, and F. Ferndandez, 'Genetic Algorithm Approach to Polyphonic Music Transcription', 2007, pp. 1–6.

[49]
H. Papadopoulos and G. Peeters, 'Large-Scale Study of Chord Estimation Algorithms Based on Chroma Representation and HMM', in International Workshop on Content-Based Multimedia Indexing, CBMI '07, Bordeaux, France, June 25-27, 2007, 2007, pp. 53–60.

[50]
L. W. Kong and T. Lee, 'Chord classification of multi-instrumental music using exemplar-based sparse representation', 2013, pp. 113–117.

[51]
K. O'Hanlon, S. Ewert, J. Pauwels, and M. B. Sandler, 'Improved template based chord recognition using the CRP feature', in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017, New Orleans, LA, USA, March 5-9, 2017, 2017, pp. 306–310.

[52]
L. Oudre, Y. Grenier, and C. Fevotte, 'Chord Recognition by Fitting Rescaled Chroma Vectors to Chord Templates', IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, pp. 2222–2233, 2011.

[53]
E. J. Humphrey, T. Cho, and J. P. Bello, 'Learning a robust Tonnetz-space transform for automatic chord recognition', 2024, pp. 453–456.

[54]
A. Uemura and J. Katto, 'Chord recognition using Doubly Nested Circle of Fifths', 2012, pp. 449–452.

[55]
F. Korzeniowski and G. Widmer, 'On the Futility of Learning Complex Frame-Level Language Models for Chord Recognition', in AES International Conference Semantic Audio 2017, Erlangen, Germany, June 22-24, 2017, 2017.

[56]
S. Liu, 'Music Tutor: Application of Chord Recognition in Music Teaching', in 2021 International Conference on Signal Processing and Machine Learning (CONF-SPML), 2021, pp. 154–157.

[57]
T. Hori, K. Nakamura, and S. Sagayama, 'Music chord recognition from audio data using bidirectional encoder-decoder LSTMs', in 2017 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference, APSIPA ASC 2017, Kuala Lumpur, Malaysia, December 12-15, 2017, 2017, pp. 1312–1315.

[58]
H.-T. Cheng, Y.-H. Yang, Y.-C. Lin, I.-B. Liao, and H. H. Chen, 'Automatic chord recognition for music classification and retrieval', 2008, pp. 1505–1508.

[59]
P. Mabpa, T. Sapaklom, E. Mujjalinvimut, J. Kunthong, and P. N. N. Ayudhya, 'Automatic Chord Recognition Technique for a Music Visualizer Application', 2021, pp. 416–419.

[60]
F. Korzeniowski and G. Widnaer, 'Automatic Chord Recognition with Higher-Order Harmonic Language Modelling', 2018, pp. 1900–1904.

[61]
J. A. Burgoyne, J. Wild, and I. Fujinaga, 'An Expert Ground Truth Set for Audio Chord Recognition and Music Analysis', in Proceedings of the 12th International Society for Music Information Retrieval Conference, ISMIR 2011, Miami, Florida, USA, October 24-28, 2011, 2011, pp. 633–638.

[62]
M. Mauch and S. Dixon, 'Approximate Note Transcription for the Improved Identification of Difficult Chords', in Proceedings of the 11th International Society for Music Information Retrieval Conference, ISMIR 2010, Utrecht, Netherlands, August 9-13, 2010, 2010, pp. 135–140.

[63]
S. H. Nawab, S. A. Ayyash, and R. Wotiz, 'Identification of musical chords using CONSTANT-Q spectra', 2001, vol. 5, pp. 3373–3376 vol.5.
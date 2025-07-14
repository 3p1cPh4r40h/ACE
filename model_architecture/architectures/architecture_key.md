# LSTM Multi Dilation Architecture Overview:

## Base Architecture Components:
- LSTM Layer - Processes the spectrogram row by row to capture temporal dependencies
- Multi-Dilation Blocks - Apply parallel dilated convolutions with various attention mechanisms
- Pooling Layer - MaxPool2d to reduce spatial dimensions
- Convolutional Layers - Three additional conv layers with batch normalization
- Fully Connected Layers - Final classification layers
- Softmax Output to Single Class (implicit from loss function)

## Architecture Variations:

### lstm_multi_dilation_A
- **Dilation Rates**: [1, 2, 4]
- **Attention**: None
- **Description**: Basic version with small dilation rates and no attention mechanism

### lstm_multi_dilation_B
- **Dilation Rates**: [2, 4, 8]
- **Attention**: Early
- **Description**: Medium dilation rates with early attention for feature selection

### lstm_multi_dilation_C
- **Dilation Rates**: [4, 8, 16]
- **Attention**: Mid
- **Description**: Large dilation rates with mid-attention for intermediate feature processing

### lstm_multi_dilation_D
- **Dilation Rates**: [8, 16, 32]
- **Attention**: Late
- **Description**: Very large dilation rates with late attention for final feature refinement

### lstm_multi_dilation_E
- **Dilation Rates**: [1, 3, 6]
- **Attention**: Softmax
- **Description**: Small dilation rates with softmax attention for probabilistic feature weighting

### lstm_multi_dilation (Original)
- **Dilation Rates**: [1, 2, 4]
- **Attention**: None
- **Description**: Original implementation with basic dilation rates and no attention
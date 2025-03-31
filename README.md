# ACE (Automated Chord Extraction)

## Instructions to Run
This code requires pytorch, pandas, scikit-learn, thop and torchinfo libraries. You will also need the **chordino features** and **MIREX labels** files from the [McGill Billboard Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/) which consists of 890 songs.

Place the folders for both the features and labels in the `data` folder in the root project directory. The `process_data.py` file will process the data into a `full_data.pkl` file containing all of the data needed for training; the `clean_data.py` file can be ran afterwards to create the `test_dataloader.pkl`, `train_dataloader.pkl` and `val_dataloader.pkl`.

Note that running `process_data.py` takes a long time; however, `clean_data.py` is much shorter and easier to modify once you have processed the data.

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

## Tests and results
The [McGill Billboard Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/) has 4 sets of labels it provides, these are the Major/Minor dataset, Major/Minor/Sevenths dataset, Major/Minor/Inversions dataset and the Major/Minor/Sevenths/Inversions dataset; these datasets have 37, 84, 90 and 213 labels respectively, including an X label to represent no chord values (these are stored as `None`, `"N"` and `"X"` in the original dataset and simplified to just `"X"` in the `clean_data.py` file). For our purposes we will refer to the datasets as M1, M2, M3 and M4 respectively, working from a the base model provided by Carsault [(a different paper)](https://doi.org/10.48550/arXiv.1911.04973).

### [Base model](10.3390/electronics10212634) performance

| Dataset  | Accuracy | F1 Score | GFLOPS | Params |
| --- | --- | --- | --- | --- |
| M1  | 0.3177%  | 0.3057  | 0.0019  | 264541
| M2  | X%  | F1  | GFLOP  | Params
| M3  | X%  | F1  | GFLOP  | Params
| M4  | X%  | F1  | GFLOP  | Params

### Base model with dilation
| Dataset  | Accuracy | F1 Score | GFLOPS | Params | Effectiveness |
| --- | --- | --- | --- | --- | --- |
| M1  | X%  | F1  | GFLOP  | Params | Effective
| M2  | X%  | F1  | GFLOP  | Params | Effective
| M3  | X%  | F1  | GFLOP  | Params | Effective
| M4  | X%  | F1  | GFLOP  | Params | Effective

## Immediate Goals
-  Find a train/test/val split based on literature
- CNN-LSTM from the [Caursault et al 2024](10.3390/electronics10212634) paper.
- Show 10 examples for each class randomly pulled from training and testing sets (and visualize how they were generated).
# ACE (Automated Chord Extraction)

# Instructions to Run
This code requires pytorch, pandas, scikit-learn, thop and torchinfo libraries. You will also need the **chordino features** and **MIREX labels** files from the [McGill Billboard Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/).

Place the folders for both the features and labels in the `data` folder in the root project directory. The `process_data.py` file will process the data into a `full_data.pkl` file containing all of the data needed for training; the `clean_data.py` file can be ran afterwards to create the `test_dataloader.pkl`, `train_dataloader.pkl` and `val_dataloader.pkl`.
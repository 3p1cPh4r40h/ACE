# ACE (Automated Chord Extraction)

## Instructions to Run
This code requires pytorch, pandas, scikit-learn, thop and torchinfo libraries. You will also need the **chordino features** and **MIREX labels** files from the [McGill Billboard Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/) which consists of 890 songs.

Place the folders for both the features and labels in the `data` folder in the root project directory. The `process_data.py` file will process the data into a `full_data.pkl` file containing all of the data needed for training; the `clean_data.py` file can be ran afterwards to create the `test_dataloader.pkl`, `train_dataloader.pkl` and `val_dataloader.pkl`.

Note that running `process_data.py` takes a long time; however, `clean_data.py` is much shorter and easier to modify once you have processed the data.

## Instructions to Add New Model
Models are stored in the `model_architecture` folder under `architectures`. To add a new model:
1. The architecture should be definined similarly to `carsault.py` (or `semi_supervised.py` if using semi-supervised techniques).
2. The model should be imported into `main.py` (see line 16 for an example).
3. The model (along with if it requires pretraining or not) should be defined in `run_models.py` in the `model_types` dict on line 12.

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
The [McGill Billboard Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/) has 4 sets of labels it provides, these are the Major/Minor dataset, Major/Minor/Sevenths dataset, Major/Minor/Inversions dataset and the Major/Minor/Sevenths/Inversions dataset; these datasets have 37, 84, 90 and 213 labels respectively, including an X label to represent no chord values (these are stored as `None`, `"N"` and `"X"` in the original dataset and simplified to just `"X"` in the `clean_data.py` file). For our purposes we will refer to the datasets as M1, M2, M3 and M4 respectively, working from a the base model provided by Carsault [(a different paper)](https://doi.org/10.48550/arXiv.1911.04973).

### [Base model](10.3390/electronics10212634) performance

| Dataset  | Accuracy | F1 Score | GFLOPS | Params |
| --- | --- | --- | --- | --- |
| M1  | 31.77%  | 0.3057  | 0.0019  | 264,541
| M2  | X%  | F1  | GFLOP  | Params
| M3  | X%  | F1  | GFLOP  | Params
| M4  | X%  | F1  | GFLOP  | Params

### Base model with dilation
| Dataset  | Accuracy | F1 Score | GFLOPS | Params | Effectiveness |
| --- | --- | --- | --- | --- | --- |
| M1  | 33.06%  | 0.3346  | 1.7435  | 32,260,328 | 0.07715
| M2  | X%  | F1  | GFLOP  | Params | Effective
| M3  | X%  | F1  | GFLOP  | Params | Effective
| M4  | X%  | F1  | GFLOP  | Params | Effective

## Immediate Goals
-  Find a train/test/val split based on literature
- CNN-LSTM from the [Caursault et al 2024](10.3390/electronics10212634) paper.
- Show 10 examples for each class randomly pulled from training and testing sets (and visualize how they were generated).
- Try using concurrent dilation layers with different rates (as opposed to a single rate and feeding dilation layers into each other). Can feed into your attention mechanism.
- Moving around the attention mechnism (early, mid and late attention).

## References

1. Bhardwaj, S., Salim, S. M., Khan, D. T. A., & JavadiMasoudian, S. (2022). Automated Music Generation using Deep Learning. pp. 193–198.

2. Choi, J., & Lee, K. (2023). Pop2Piano: Pop Audio-Based Piano Cover Generation. pp. 1-5.

3. Wu, Y., Carsault, T., & Yoshii, K. (2019). Automatic Chord Estimation Based on a Frame-wise Convolutional Recurrent Neural Network with Non-Aligned Annotations. pp. 1-5.

4. Yamaga, H., Momma, T., Kojima, K., & Itoh, Y. (2023). Ensemble of Transformer and Convolutional Recurrent Neural Network for Improving Discrimination Accuracy in Automatic Chord Recognition. pp. 2299–2305.

5. Humphrey, E. J., & Bello, J. P. (2012). Rethinking Automatic Chord Recognition with Convolutional Neural Networks. vol. 2, pp. 357–362.

6. Maruo, S., Yoshii, K., Itoyama, K., Mauch, M., & Goto, M. (2015). A feedback framework for improved chord recognition based on NMF-based approximate note transcription. pp. 196–200.

7. Durán, G., & de la Cuadra, P. (2021). Transcribing Lead Sheet-Like Chord Progressions of Jazz Recordings. Computer Music Journal, vol. 44, pp. 26–42.

8. Vaca, K., Gajjar, A., & Yang, X. (2019). Real-Time Automatic Music Transcription (AMT) with Zync FPGA. pp. 378–384.

9. Shibata, K., et al. (2019). Joint Transcription of Lead, Bass, and Rhythm Guitars Based on a Factorial Hidden Semi-Markov Model. pp. 236–240.

10. Pedroza, H., Abreu, W., Corey, R. M., & Roman, I. R. (2025). Guitar-TECHS: An Electric Guitar Dataset Covering Techniques, Musical Excerpts, Chords and Scales Using a Diverse Array of Hardware. pp. 1-5.

11. Sun, J., Li, H., & Lei, L. (2009). Key detection through pitch class distribution model and ANN. pp. 1-6.

12. Humphrey, E. J., & Bello, J. P. (2014). From music audio to chord tablature: Teaching deep convolutional networks to play guitar. In 2014 IEEE International Conference on Acoustic, Speech and Signal Processing (ICASSP).

13. Bortolozzo, M., Schramm, R., & Jung, C. R. (2021). Improving the Classification of Rare Chords With Unlabeled Data. In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

14. Carsault, T., Nika, J., Esling, P., & Assayag, G. (2021). Combining Real-Time Extraction and Prediction of Musical Chord Progressions for Creative Applications. vol. 10.

15. Jadhav, Y., Patel, A., Jhaveri, R. H., & Raut, R. (2022). Transfer Learning for Audio Waveform to Guitar Chord Spectrograms Using the Convolution Neural Network. Mobile Information Systems, vol. 2022, pp. 1–11.

16. Lee, Y.-S., Chiang, Y.-L., Lin, P.-R., Lin, C.-H., & Tai, T.-C. (2016). Robust and efficient content-based music retrieval system. APSIPA Transactions on Signal and Information Processing, vol. 5, no. 1.

17. Li, N. (2022). Generative Adversarial Network for Musical Notation Recognition during Music Teaching. Computational Intelligence and Neuroscience, vol. 2022, pp. 1–9.

18. March, Pattaya, & Thailand. (2021). Automatic Chord Recognition Technique for a Music Visualizer Application. In 2021 9th International Electrical Engineering Congress (iEECON).

19. Mandel, E. (2021). Chord Recognition From DFTs of Down Sampled Audio Signals. In 2021 IEEE Western New York Image and Signal Processing Workshop (WNYISPW).

20. Mukherjee, H., et al. (2020). Music chord inversion shape identification with LSTM-RNN. Procedia Computer Science, vol. 167, pp. 607–615.

21. Nishikimi, R., Nakamura, E., Goto, M., & Yoshii, K. (2021). Audio-to-score singing transcription based on a CRNN-HSMM hybrid model. APSIPA Transactions on Signal and Information Processing, vol. 10, no. 1.

22. Ojima, Y., Nakamura, E., Itoyama, K., & Yoshii, K. (2018). Chord-aware automatic music transcription based on hierarchical Bayesian integration of acoustic and language models. APSIPA Transactions on Signal and Information Processing, vol. 7, no. 1.

23. Pauwels, J., Martens, J.-P., & Leman, M. (2011). The Influence of Chord Duration Modeling on Chord and Local Key Extraction. In 2011 10th International Conference on Machine Learning and Applications and Workshops, pp. 136–141.

24. Perez, A., Ma, H. L., Zawaduk, S., & Dawson, M. R. W. (2023). How Do Artificial Neural Networks Classify Musical Triads? A Case Study in Eluding Bonini's Paradox. Cognitive Science, vol. 47, no. 1.

25. de Berardinis, J., Meroño-Peñuela, A., Poltronieri, A., & Presutti, V. (2023). ChoCo: a Chord Corpus and a Data Transformation Workflow for Musical Harmony Knowledge Graphs. Scientific Data, vol. 10, no. 1.

26. Rida, I., Herault, R., & Gasso, G. (2014). Supervised Music Chord Recognition. In 2014 13th International Conference on Machine Learning and Applications, pp. 336–341.

27. Shi, S., Xi, S., & Tsai, S.-B. (2021). Research on Autoarrangement System of Accompaniment Chords Based on Hidden Markov Model with Machine Learning. Mathematical Problems in Engineering, vol. 2021, pp. 1–10.

28. Wang, W. (2023). Music chord sequence recognition method based on audio feature extraction algorithm. In 2023 IEEE International Conference on Control, Electronics and Computer Technology (ICCECT).

29. Wu, Y., & Yoshii, K. (2022). Joint Chord and Key Estimation Based on a Hierarchical Variational Autoencoder with Multi-task Learning. APSIPA Transactions on Signal and Information Processing, vol. 11, no. 1.

30. Yu, Y., Zimmermann, R., Wang, Y., & Oria, V. (2012). Recognition and Summarization of Chord Progressions and Their Application to Music Information Retrieval. In 2012 IEEE International Symposium on Multimedia, pp. 9–16.

31. Zhuang, C. (2021). GCA:A chord music generation algorithm based on double-layer LSTM. In 2021 3rd International Conference on Advances in Computer Technology, Information Science and Communication (CTISC).

32. Ashley, J., et al. (2011). AN EXPERT GROUND-TRUTH SET FOR AUDIO CHORD RECOGNITION AND MUSIC ANALYSIS. In 12th International Society for Music Information Retrieval Conference (ISMIR 2011).

33. Korzeniowski, F., Widmer, G., & Institute of Computational Perception. (2018). Automatic Chord Recognition with Higher-Order Harmonic Language Modelling. In 2018 26th European Signal Processing Conference (EUSIPCO).

34. Brunner, G., Wang, Y., Wattenhofer, R., & Wiesendanger, J. (2017). JamBot: Music Theory Aware Chord Based Generation of Polyphonic Music with LSTMs. In 2017 IEEE 29th International Conference on Tools with Artificial Intelligence (ICTAI), pp. 519–526.

35. Gagnon, T., Larouche, S., & Lefebvre, R. (2003). A neural network approach for preclassification in musical chords recognition. vol. 2, pp. 2106-2109 Vol.2.

36. Korzeniowski, F., Sears, D. R. W., & Widmer, G. (2018). A Large-Scale Study of Language Models for Chord Prediction. CoRR, vol. abs/1804.01849.

37. Wu, Y., Nakamura, E., & Yoshii, K. (2020). A Variational Autoencoder for Joint Chord and Key Estimation from Audio Chromagrams. pp. 500–506.

38. Mauch, M., Dixon, S., Queen Mary University of London, & Centre for Digital Music. (2010). APPROXIMATE NOTE TRANSCRIPTION FOR THE IMPROVED IDENTIFICATION OF DIFFICULT CHORDS. In Proceedings of the 11th International Society for Music Information Retrieval Conference, ISMIR 2010, Utrecht, Netherlands, August 9-13, 2010, pp. 135–140.

39. Eremenko, V., Demirel, E., Bozkurt, B., & Serra, X. (2018). Audio-Aligned Jazz Harmony Dataset for Automatic Chord Transcription and Corpus-based Research. In Proceedings of the 19th International Society for Music Information Retrieval Conference, ISMIR 2018, Paris, France, September 23-27, 2018, pp. 483–490.

40. Orio, N. (2006). Music Retrieval: A Tutorial and Review. Found. Trends Inf. Retr., vol. 1, no. 1, pp. 1–90.

41. Wang, Z., et al. (2020). POP909: A Pop-Song Dataset for Music Arrangement Generation. In Proceedings of the 21th International Society for Music Information Retrieval Conference, ISMIR 2020, Montreal, Canada, October 11-16, 2020, pp. 38–45.

42. Oudre, L., Fevotte, C., & Grenier, Y. (2011). Probabilistic Template-Based Chord Recognition. IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, pp. 2249–2259.

43. R. M., G., Rao, K. S., & Das, P. P. (2018). Harmonic-Percussive Source Separation of Polyphonic Music by Suppressing Impulsive Noise Events. In 19th Annual Conference of the International Speech Communication Association, Interspeech 2018, Hyderabad, India, September 2-6, 2018, pp. 831–835.

44. Sleep, J. (2017). Automatic Music Transcription with Convolutional Neural Networks using Intuitive Filter Shapes. Robert E. Kennedy Library, Cal Poly.

45. Brown, J. C. (1991). Calculation of a constant Q spectral transform. The Journal of the Acoustical Society of America, vol. 89, no. 1, pp. 425–434.

46. Wu, Y., Carsault, T., Nakamura, E., & Yoshii, K. (2020). Semi-Supervised Neural Chord Estimation Based on a Variational Autoencoder With Latent Chord Labels and Features. IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2956–2966.

47. Yu, Y., Zimmermann, R., Wang, Y., & Oria, V. (2013). Scalable Content-Based Music Retrieval Using Chord Progression Histogram and Tree-Structure LSH. IEEE Transactions on Multimedia, vol. 15, pp. 1969–1981.

48. Reghunath, L. C., & Rajan, R. (2023). Predominant audio source separation in polyphonic music. EURASIP Journal on Audio, Speech, and Music Processing, vol. 2023, no. 1.

49. Tverdokhleb, E., Myronova, N., & Fedoronchak, T. (2017). Music signal processing to obtain its chorded representation. pp. 301–304.

50. Deng, J.-Q., & Kwok, Y.-K. (2017). LARGE VOCABULARY AUTOMATIC CHORD ESTIMATION WITH AN EVEN CHANCE TRAINING SCHEME. In Proceedings of the 18th International Society for Music Information Retrieval Conference, ISMIR 2017, Suzhou, China, October 23-27, 2017, pp. 531–536.

51. Koops, H. V., de Haas, W. B., Bransen, J., & Volk, A. (2020). Automatic chord label personalization through deep learning of shared harmonic interval profiles. Neural Comput. Appl., vol. 32, no. 4, pp. 929–939.

52. Row, E., Tang, J., & Fazekas, G. (2023). JAZZVAR: A Dataset of Variations found within Solo Piano Performances of Jazz Standards for Music Overpainting. CoRR, vol. abs/2307.09670.

53. Reis, G., Fonseca, N., & Ferndandez, F. (2007). Genetic Algorithm Approach to Polyphonic Music Transcription. pp. 1-6.

54. Wotiz, S. H. N. A. A. (2001). Identification of musical chords using CONSTANT-Q spectra. vol. 5, pp. 3373–3376 vol.5.

55. Papadopoulos, H., & Peeters, G. (2007). Large-Scale Study of Chord Estimation Algorithms Based on Chroma Representation and HMM. In International Workshop on Content-Based Multimedia Indexing, CBMI '07, Bordeaux, France, June 25-27, 2007, pp. 53–60.

56. Kong, L. W., & Lee, T. (2013). Chord classification of multi-instrumental music using exemplar-based sparse representation. pp. 113–117.

57. O'Hanlon, K., Ewert, S., Pauwels, J., & Sandler, M. B. (2017). Improved template based chord recognition using the CRP feature. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017, New Orleans, LA, USA, March 5-9, 2017, pp. 306–310.

58. Oudre, L., Grenier, Y., & Fevotte, C. (2011). Chord Recognition by Fitting Rescaled Chroma Vectors to Chord Templates. IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, pp. 2222–2233.

59. Humphrey, E. J., Cho, T., & Bello, J. P. (2024). Learning a robust Tonnetz-space transform for automatic chord recognition. pp. 453–456.

60. Uemura, A., & Katto, J. (2012). Chord recognition using Doubly Nested Circle of Fifths. pp. 449–452.

61. Korzeniowski, F., & Widmer, G. (2017). On the Futility of Learning Complex Frame-Level Language Models for Chord Recognition. In AES International Conference Semantic Audio 2017, Erlangen, Germany, June 22-24, 2017.

62. Liu, S. (2021). Music Tutor: Application of Chord Recognition in Music Teaching. In 2021 International Conference on Signal Processing and Machine Learning (CONF-SPML), pp. 154–157.

63. Hori, T., Nakamura, K., & Sagayama, S. (2017). Music chord recognition from audio data using bidirectional encoder-decoder LSTMs. In 2017 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference, APSIPA ASC 2017, Kuala Lumpur, Malaysia, December 12-15, 2017, pp. 1312–1315.

64. Cheng, H.-T., Yang, Y.-H., Lin, Y.-C., Liao, I.-B., & Chen, H. H. (2008). Automatic chord recognition for music classification and retrieval. pp. 1505–1508.
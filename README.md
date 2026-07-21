# Decoding EEG-Based Movement Imagery (MI) using a CNN-Transformer

This project attempts to develop a CNN-Transformer, combining scientific techniques which have proven highly useful in research the past couple of years, which can decode movement imagination from EEG data reliably and hence outperform other state-of-the-art models.

## Literature Background
##### Brain Computer Interfaces (BCIs)
Brain Computer Interfaces (BCIs) enable the control of external devices or systems solely through measured brain activity (Liao et al., 2025). BCIs can therefore be used for prosthetic limb operations or in stroke patient mobilization (Liao et al., 2025). A promising BCI application, which inspired this project, is an EEG-based BCI for limb movement based on movement imagination, for instance for patients with phantom limbs. 

BCIs typically first acquire data from either noninvasive skull measurements, using EEG, or invasive methods like ECoG. The signals are then processed, relevant features extracted, classified and the feedback is then used to optimize the procedure. Especially in spinal cord injuries, where the communication of muscle spindles with higher cognitive areas is impaired, represents a motivation to work on such devices. 

##### Deep Learning and BCIs
Different deep learning (DL) models have been applied to decode movement imagination (MI), among those being CNNs, RNNs, LSTMs and GRUs, as well as Transformers. While CNNs are good at local feature extraction, they struggle with global information. For Transformers, the opposite is the case: Since they have no inductive local bias, they require large datasets to converge. However, EEG data is typically quite scarce and hence a Transformer might potentially overfit the data. 

##### Challenges of EEG Data
EEG data is usually highly variable, non-stationary, typically scarce, and has a low signal-to-noise ratio (Liao et al., 2025). To tackle those challenges, research has focused on CNN-Transformers in recent years, combining the local strength of CNNs with global focus of a Transformer.

## Project
The project aims to build on Ma et al. (2022) and Liao et al. (2025), but inspiration from further recent scientific papers and publications has been drawn to improve the model's accuracy as well.

Kavira & Vinjamuri (2025) have highlighted the importance of advanced preprocessing of EEG data. Therefore, in this project band-pass filtering using ICA decomposition and feature extraction via PSD has been used.

In the *CNN* a dropout ratio of 0.3 and weight decay of 0.5 are used, to prevent overfitting (inspired by Liao et al. (2025)).

The *Transformer* is built on an attention layer and feed-forward network, followed by a normalization layer and a swish gated linear unit (SwiGLU) as activation function, which has proven very useful in the paper from Liao et al. (2025).

For the *Training* categorical cross entropy with label smoothing has been used (again inspired by Liao et al. (2025)). 

As *Validation* matrices, accuracy, the amount of correctly classified samples, and Cohen's kappa are used. Cohen's kappa is especially useful for scenarios with imbalanced data distributions (Liao et al., 2025). To compare the classification results for my model to other state-of-the-art models, confusion matrices have been computed ()

#### Limitations
Gao & Diniz (2026) demonstrated that the EEG decoding architecture inductive bias might matter more than the raw data volume, illustrating the importance of a good architectural choice.

#### Extensions / Discussion
Nice extensions for this project could be using a Temporal Convolutional Network (TCN) as Liao et al. (2025) for time-series modeling and capturing nuanced temporal dynamics. To handle data nonstationarity, multivariate empirical mode decomposition (MEMD) could also improve the model's results potentially (Liao et al., 2025). Further, Kaviri & Vinjamuri (2025) have illustrated the high potential of source localization techniques, like beaqm forcing, to compare accuracy scores. Another extension could be applying Grad-CAM or gate-weight analyses which links prediction to oscillatory bands for interpretation, as done by Gao & Diniz (2026), who also suggest implementing zero-shot transfer learning. Besides that, spherical positional encoding, as suggested by Yuce & Stober (2026), could represent a fruitful extension.

### References
* Gao, Y., & Diniz, J. M. (2026). From perception to imagination: a robust deep learning architecture for real-time EEG decoding and neurophysiological validation. Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization, 14(1). https://doi.org/10.1080/21681163.2026.2684109.
* Kaviri, S. M., & Vinjamuri, R. (2025). Decoding motor execution and motor imagery from EEG with deep learning and source localization. Biomedical Engineering Advances,
9, 100156: 2667-0992.https://doi.org/10.1016/j.bea.2025.100156.
* Liao, W., Liu, H. & Wang, W. (2025). Advancing BCI with a transformer-based model for motor imagery classification. Sci Rep 15, 23380. https://doi.org/10.1038/s41598-025-06364-4. - the code is available here: https://github.com/BlackCattt9/EEGEncoder
* Ma, Y., Song, Y. & Gao, F. (2022). A novel hybrid CNN-Transformer model for EEG Motor Imagery classification. International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, 1-8. https://doi.org/10.1109/IJCNN55064.2022.9892821.
* Yuce, A. B., & Stober, S. (2026). Benchmarking Positional Encoding Strategies for Transformer-Based EEG Foundation Models [Arxiv Preprint].     
https://doi.org/10.48550/arXiv.2605.29754.
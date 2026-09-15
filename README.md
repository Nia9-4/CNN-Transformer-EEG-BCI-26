# Decoding EEG-Based Movement Imagery (MI) using a CNN-Transformer
_by Antonia Reul (2026)_ | Contact: areul@uni-osnabrueck.de

This project attempts to develop a CNN-Transformer, combining scientific techniques which have proven highly useful in research the past couple of years, which can decode movement imagination from EEG data reliably.

Please see the PDF **'decoding_eeg_movement_imagery.pdf'** for a **full report on architectural and parameter choices**. 

## Setup
The MNE library, as well as other medical libraries, are still getting upgraded and unfortunately sometimes still rely on old code. To ensure that the code works, the following installations are recommended:

* python = 3.10 - MNE is more stable on it than on 3.12
* numpy < 2.0 - MNE relies on a function which has been removed and renamed in NumPy2.0
* scipy >= 1.11 

        Users are encouraged to ensure a proper running of the 'pipeline_test.ipynb' to make sure, the main code can be properly executed.


_The original idea of this project was to build on Ma et al. (2022) and Liao et al. (2025), but inspiration from further recent scientific papers and publications has been drawn as well._

### References
* Liao, W., Liu, H. & Wang, W. (2025). Advancing BCI with a transformer-based model for motor imagery classification. Sci Rep 15, 23380. https://doi.org/10.1038/s41598-025-06364-4. - the code is available here: https://github.com/BlackCattt9/EEGEncoder
* Ma, Y., Song, Y. & Gao, F. (2022). A novel hybrid CNN-Transformer model for EEG Motor Imagery classification. International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, 1-8. https://doi.org/10.1109/IJCNN55064.2022.9892821.
* Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C28G6P - documentation: https://physionet.org/content/eegmmidb/1.0.0/

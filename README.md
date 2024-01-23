# BAFLineDP: Code Bilinear Attention Fusion Framework for Line-level Defect Prediction
This paper has been accepted for inclusion in the SANER 2024.

## Environment Setup
### Python Environment Setup
The implementation codes are running in the following environment setups.
- `python == 3.9.13`
- `pytorch == 1.12.1`
- `numpy == 1.23.2`
- `pandas == 1.5.3`
- `transformers == 4.20.1`
- `joblib == 1.2.0`
- `more-itertools == 9.1.0`
- `rdkit == 2023.3.1`
- `scikit-learn == 1.1.2`
- `tqdm == 4.64.1`

### R Environment Setup
Download the following packages: `tidyverse`, `gridExtra`, `ModelMetrics`, `caret`, `reshape2`, `pROC`, `effsize`, and `progress`.

## Experiment
### Experimental Setup
The following parameters are used to train our BAFLineDP model.
- `batch_size` = 16
- `num_epochs` = 10
- `embed_dim` = 768
- `gru_hidden_dim` = 64
- `gru_num_layers` = 1
- `bafn_hidden_dim` = 256
- `dropout` = 0.2
- `lr (learning rate)` = 0.001

### Code Preprocessing
Download the datasets from the [github_url](https://github.com/awsm-research/line-level-defect-prediction) and keep them in `datasets/original/`.

Run the following command to prepare data for file-level model training. The output will be stored in `datasets/preprocessed_data/`.

	python code_preprocessing.py

### BAFNLineDP Model Training
To train BAFNLineDP model for each project, run the following command. The trained models will be saved in `output/model/BAFLineDP/`, and the loss will be saved in `output/loss/BAFLineDP/`.

	python train_model.py

### Prediction Generation
To make a prediction within software projects, run the following command. 

	python generate_within_prediction.py

The generated outputs of within-prediction are stored in `output/prediction/BAFLineDP/within-release/`.

To make a prediction across software projects, run the following command.
	
	python generate_cross_prediction.py

The generated outputs of cross-prediction are stored in `output/prediction/BAFLineDP/cross-release/`.

### Get the Evaluation Results
To obtain file-level and line-level defect prediction results within WPDP and CPDP scenarios, you need to set the absolute working path for the script to run first, and then execute the following command.

	Rscript  get_results.R

The results are stored in `output/result/BAFLineDP/`.

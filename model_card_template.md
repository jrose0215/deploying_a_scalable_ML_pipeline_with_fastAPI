# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained using scikit-learn (version 1.5.1) 
with 100 estimators and a random state of 42. It was developed as part of an MLOps 
pipeline project to predict whether an individual earns more or less than $50K per year 
based on census data.

## Intended Use

This model is intended to predict income levels (>50K or <=50K) from demographic and 
employment-related census data. It is designed for educational and demonstration 
purposes as part of an end-to-end MLOps pipeline. It should not be used for making 
real-world financial or employment decisions.

## Training Data

The model was trained on the Census Income dataset from the UCI Machine Learning 
Repository. The dataset contains 32,561 records with 14 features including age, 
workclass, education, marital status, occupation, relationship, race, sex, 
capital gain, capital loss, hours per week, and native country. The data was split 
80/20 into training and test sets using a random state of 42. Categorical features 
were encoded using a OneHotEncoder and the target label was binarized using a 
LabelBinarizer.

## Evaluation Data

The evaluation data consists of 20% of the original Census Income dataset, 
approximately 6,513 records, held out from training using a fixed random split. 
The same preprocessing pipeline applied to the training data was applied to the 
evaluation data using the fitted encoder and label binarizer.

## Metrics

The model was evaluated using precision, recall, and F1 score. On the test dataset 
the model achieved the following performance:

- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

Performance was also computed on slices of the data across all categorical features. 
For example, for the workclass feature, precision ranged from 0.65 to 1.00 depending 
on the category, reflecting variation in model performance across different subgroups.

## Ethical Considerations

The Census Income dataset contains sensitive demographic attributes including race, 
sex, and native country. The model may reflect historical biases present in the data. 
Performance varies across demographic slices, which should be carefully considered 
before any real-world application. This model should not be used to make decisions 
that could negatively impact individuals based on protected characteristics.

## Caveats and Recommendations

The model was trained on census data from 1994 and may not reflect current income 
distributions or workforce demographics. The model is intended for educational 
purposes only. Before any production use, the model should be retrained on more 
recent data and thoroughly evaluated for fairness across all demographic groups. 
Additionally, hyperparameter tuning and cross-validation could be explored to 
improve model performance.
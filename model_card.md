# Model Card

## Model Details
This model is a pre-trained Random Forest classifier implemented using the scikit-learn framework. It is designed to predict whether the salary of an individual or a group of individuals exceeds or falls below $50,000, based on features derived from the U.S. Census Income Data Set. The model was trained utilizing cross-validation techniques to ensure robust performance.

Developed by Alex Spiers as part of the Udacity Nanodegree for Machine Learning DevOps Engineer program.

## Intended Use
The model is intended to be deployed on Heroku as a web application and integrated with an API developed using FastAPI. The API is currently live in production within a comprehensive CI/CD framework.

While this project serves as a valuable portfolio piece, it can also be adapted for use in various projects, particularly those focused on continuous integration and deployment. Importantly, the model is not intended for estimating individual salaries or informing pay benchmarking. Instead, it showcases the author’s skills in MLOps, particularly in the context of continuous integration.

## Training Data
The training data used for this model is the U.S. Census Income Data Set, which is accessible through the UCI Machine Learning Repository. For model training, 80% of the data was allocated for cross-validation.

## Evaluation Data
A hold-out dataset comprising 20% of the total data was used for evaluation purposes.

## Metrics
The evaluation metrics utilized to assess model performance include:

- **Accuracy**: Overall correctness of the model’s predictions.
- **Precision**: Calculated as \(\text{Precision} = \frac{TP}{TP + FP}\), where TP represents true positives and FP represents false positives.
- **Recall**: Defined as \(\text{Recall} = \frac{TP}{TP + FN}\), where FN represents false negatives.
- **F1 Score**: The harmonic mean of Precision and Recall.
- **AUC of ROC**: The Area Under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between classes.

### Model Performance on Test Set
- Test Set Accuracy: **0.855**
- Test Set F1 Score: **0.807**
- Test Set Precision: **0.526**
- Test Set Recall: **0.600**
- Test Set AUC: **0.743**

For a detailed performance analysis on Education, Sex, and Race subcategories, please refer to `slice_output.txt`.

## Ethical Considerations
### Data
The dataset contains sensitive attributes such as race, education, and income information. Fortunately, it has been anonymized, reducing the risk of participant identification. However, certain demographic classes, particularly related to race or country of origin, are underrepresented, which may introduce representation bias. Consequently, models trained on this data might reflect biases toward groups that are over- or under-represented relative to their proportions in the U.S. population.

### Usage Risks and Harms
There is potential for misuse of this model in contexts such as hiring or salary benchmarking. To address these concerns, data slicing techniques were employed to investigate and assess model bias.

## Caveats and Recommendations
While the model achieves a reasonable accuracy, its performance is generally limited. As a binary classifier predicting salary thresholds (above or below $50,000), it has restricted applicability. A dataset with continuous income information would enhance its utility and predictive power.

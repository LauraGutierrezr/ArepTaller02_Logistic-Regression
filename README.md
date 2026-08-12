# Heart Disease Risk Prediction – Logistic Regression

This project implements **logistic regression from first principles** to predict the risk of heart disease using real-world clinical data. The model is built without high-level machine learning libraries and focuses on understanding the full pipeline: data exploration, model training, evaluation, regularization, visualization, and cloud deployment.  
The project is developed as part of a Machine Learning assignment, emphasizing both theoretical foundations and practical deployment in cloud environments.

## Getting Started

These instructions will help you set up the project locally for development and experimentation. Instructions for cloud execution using **AWS SageMaker** are included in the Deployment section.

### Prerequisites

You need the following software installed:

- Python 3.9 or later  
- Jupyter Notebook or Jupyter Lab  

Allowed Python libraries (as specified in the assignment):

```
numpy
pandas
matplotlib
```

High-level machine learning libraries such as scikit-learn, TensorFlow, or PyTorch are **not used** for model training.

### Installing

Create and activate a virtual environment (optional but recommended):

```
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```
pip install numpy pandas matplotlib jupyter
```

Start Jupyter Notebook:

```
jupyter notebook
```

Open the main notebook, for example:

```
heart_disease_lr_analysis.ipynb
```

Run all cells to perform data exploration, train the logistic regression model, visualize decision boundaries, and evaluate model performance.

## Dataset

The dataset used in this project is the **Heart Disease Dataset** from Kaggle:

- **270 patient records** (after cleaning)
- **14 clinical features** including:
  - Age: 29-77 years
  - Cholesterol: 126-564 mg/dL
  - Blood Pressure: 94-200 mmHg
  - Max Heart Rate: 71-202 bpm
  - ST Depression: 0-6.2
  - Number of Vessels: 0-3
- **Binary target variable:** 1 = disease present (44.4%), 0 = disease absent (55.6%)
- **Selected features for model:** Age, Cholesterol, BP, Max HR, ST Depression, Number of Vessels

Dataset source:  
https://www.kaggle.com/datasets/neurocipher/heartdisease


### Break down into end-to-end tests

End-to-end validation consists of:

- Running all notebook cells without errors  
- Verifying that the cost function decreases over training iterations  
- Confirming reasonable accuracy, precision, recall, and F1-score on train and test sets  

PROOF OF THIS

<img width="729" height="599" alt="Captura de pantalla 2026-08-11 a la(s) 11 37 03 p m" src="https://github.com/user-attachments/assets/e91f0f84-e964-4a3c-b877-989465b795c0" />


<img width="795" height="657" alt="Captura de pantalla 2026-08-11 a la(s) 11 37 32 p m" src="https://github.com/user-attachments/assets/c49d729f-4ddb-4362-8c8d-ccd01d1bde3d" />


<img width="847" height="649" alt="Captura de pantalla 2026-08-11 a la(s) 11 37 44 p m" src="https://github.com/user-attachments/assets/8c0cebd5-5624-49c3-8d29-3259c47951ad" />


<img width="844" height="707" alt="Captura de pantalla 2026-08-11 a la(s) 11 37 53 p m" src="https://github.com/user-attachments/assets/cd99cd04-3826-4d6c-8b31-248daf7cac01" />


<img width="852" height="394" alt="Captura de pantalla 2026-08-11 a la(s) 11 38 06 p m" src="https://github.com/user-attachments/assets/07bbac18-40ee-4f65-b836-b80fe605d988" />

All tested λ values (0 to 1) maintain **85.2% test accuracy** and **82.9% F1-score**
- The model is not overfitting significantly, so regularization has minimal impact
  

### And coding style tests

Coding style validation focuses on:

- Explicit implementation of sigmoid, cost function, and gradient descent  
- Clear separation between data preprocessing, model logic, and evaluation  
- Proper use of NumPy vectorized operations  

## Deployment

The best-performing logistic regression model is exported and explored in **AWS SageMaker Studio** to simulate a production deployment scenario.

### Model Export

The model is saved as a NumPy `.npz` file containing:
- Trained weights (w)
- Bias term (b)
- Normalization parameters (mean, std)

### Deployment Steps

1. **Upload notebook to SageMaker Studio**
2. **Train model in cloud environment**
3. **Export model parameters** (`heart_disease_model.npz`)
4. **Create inference endpoint** for real-time predictions

### Sample Inference

**Input (Sample Patient):**
```json
{
  "age": 60,
  "cholesterol": 300,
  "bp": 140,
  "max_hr": 150,
  "st_depression": 2.0,
  "num_vessels": 2
}
```

**Output:**
```json
{
  "probability": 0.846,
  "prediction": "High Risk"
}
```

The model correctly identifies this patient as high risk (84.6% probability) based on elevated cholesterol, ST depression, and 2 affected vessels.

### Expected Latency

Real-time inference latency: ~50-100ms per prediction on ml.t2.medium instance.

## Built With

* Python – Core programming language  
* NumPy – Numerical computation  
* Pandas – Data manipulation  
* Matplotlib – Data visualization  
* Jupyter Notebook – Interactive development environment  
* AWS SageMaker Studio – Cloud training and deployment platform  


---
## Final Insights and Conclusions

### Summary

This notebook implements logistic regression from scratch for heart disease prediction:

1. **Data Preparation (Step 1):** Loaded 270 patient records from Kaggle, performed EDA, normalized features, and created a 70/30 stratified split maintaining class balance.

2. **Model Implementation (Step 2):** Built sigmoid, cost function (binary cross-entropy), and gradient descent from scratch using NumPy. No scikit-learn was used for training.

3. **Performance:** Achieved **85.2% accuracy** and **82.9% F1-score** on the test set with good precision/recall balance.

4. **Decision Boundaries (Step 3):** Visualized 3 feature pairs; **ST Depression + Number of Vessels** showed the best linear separability.

5. **Regularization (Step 4):** L2 regularization with λ=0.01 provides minimal improvement since the model already generalizes well.

6. **Deployment (Step 5):** Prepared model export (.npz format) and inference handler ready for Amazon SageMaker deployment.

### Key Findings

| Feature | Weight | Interpretation |
|---------|--------|----------------|
| `num_vessels` | +0.89 | More blocked vessels → Higher risk |
| `st_depression` | +0.62 | Higher ST depression → Higher risk (ischemia indicator) |
| `max_hr` | -0.65 | Lower max heart rate → Higher risk (poor fitness) |
| `bp` | +0.12 | Higher blood pressure → Slightly higher risk |
| `cholesterol` | +0.11 | Higher cholesterol → Slightly higher risk |
| `age` | -0.14 | Age effect captured by other features |

### Performance Metrics

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 76.7% | 85.2% |
| Precision | 78.6% | 85.3% |
| Recall | 65.5% | 80.6% |
| F1-Score | 71.4% | 82.9% |


### Clinical Interpretation

The model aligns with medical knowledge:
- **Number of vessels** with fluoroscopy-detected blockages is the strongest predictor
- **ST depression** during exercise stress tests indicates cardiac ischemia
- **Lower maximum heart rate** suggests poor cardiovascular fitness

### Sample Inference Test

I accessed the AWS console and opened the Amazon SageMaker Studio service. From there, I created a new notebook by selecting the Python 3 kernel, which already includes the necessary libraries such as NumPy

<img width="979" height="93" alt="Captura de pantalla 2026-08-11 a la(s) 11 18 38 p m" src="https://github.com/user-attachments/assets/e1a38484-f9a7-4622-a651-b2395def0067" />


Once the notebook was created, copy the code developed for logistic regression. This code includes the model functions, training, and prediction. All cells ran correctly within the SageMaker environment. After training the model, the final parameters were obtained: the weight vector (w) and the bias (b). These values represent the trained model and are sufficient to make predictions on new data

<img width="256" height="117" alt="Captura de pantalla 2026-08-11 a la(s) 11 20 14 p m" src="https://github.com/user-attachments/assets/bc57c445-f745-483a-967a-aef8e94159ff" />

To simulate an inference endpoint, I defined a function that receives patient data and returns the probability of heart disease using the sigmoid function. This function represents the basic behavior of a SageMaker endpoint.

The model was then tested with a sample patient using normalized feature values. When the prediction function was run, the model returned a probability of 0.81, indicating an 81% probability of heart disease.


<img width="1235" height="517" alt="Captura de pantalla 2026-08-11 a la(s) 11 12 54 p m" src="https://github.com/user-attachments/assets/e86b42a6-9f86-4ec4-9a09-c13d97fb02c0" />

- The model returns a probability rather than a class directly


- Input: Age = 60, Cholesterol = 300 (normalized input values)
- Output: Predicted probability of heart disease ≈ 0.81

Implementation in Amazon SageMaker allowed the model to be run in a managed environment based on JupyterLab, facilitating training and inference without additional configurations. The model can be used for real-time risk scoring, returning interpretable probabilities for individual patients. In tests with simulated data, the system responded immediately, showing negligible latency in the notebook environment


## Authors

* **Valentina Gutiérrez** – Initial work and implementation  

## License

This project is for academic use only as part of a university course assignment.

## Acknowledgments

* Course instructors for guidance on machine learning fundamentals  
* Kaggle for providing the Heart Disease Dataset  
* AWS Academy for cloud infrastructure support  

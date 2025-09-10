
---

# 📊 Telco Customer Churn Prediction

This project focuses on predicting **customer churn** in the telecommunications sector using **data preprocessing, feature engineering, and machine learning models**. The repository includes exploratory data analysis (EDA), model training and evaluation, and full pipeline scripts for deployment and inference.

---

## 🚀 Features

* **EDA**: Handling missing values, outliers, feature binning, encoding, and scaling.
* **Model Implementation & Evaluation**:

  * Base model training
  * K-Fold cross-validation
  * Multi-model training and comparison
  * Hyperparameter tuning
  * Threshold optimization
  * Best model selection and usage
* **Pipelines**:

  * Data pipeline
  * Training pipeline
  * Streaming inference pipeline
* **Reusable Modules**: Modularized preprocessing, feature engineering, model building, training, evaluation, and inference.

---

## 📂 Repository Structure

```
TELCO-CUSTOMER-CHURN-PREDICTION
│
├── EDA/                           # Jupyter notebooks for data preprocessing & EDA
│   ├── 0_Handling_Missing_Values.ipynb
│   ├── 1_Handling_Outliers.ipynb
│   ├── 2_Feature_Binning.ipynb
│   ├── 3_Encoding_And_Scaling.ipynb
│   └── requirments.txt
│
├── Model_Implementation_andEvaluation/   # Model training & evaluation
│   ├── 1_Base_Model_Training.ipynb
│   ├── 2_kfold_validation.ipynb
│   ├── 3_Multi_model_training.ipynb
│   ├── 4_Hyperparam_tuning.ipynb
│   ├── 5_Threshold_optimization.ipynb
│   └── 6_Using_the_Best_Model.ipynb
│
├── Python_Script/
│   ├── pipelines/                 # ML pipelines
│   │   ├── data_pipeline.py
│   │   ├── training_pipeline.py
│   │   └── streaming_inference_pipeline.py
│   │
│   ├── src/                       # Core reusable modules
│   │   ├── data_ingestion.py
│   │   ├── data_splitter.py
│   │   ├── feature_binning.py
│   │   ├── feature_encoding.py
│   │   ├── feature_scaling.py
│   │   ├── handle_missing_values.py
│   │   ├── outlier_detection.py
│   │   ├── model_building.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   └── model_inference.py
│   │
│   └── utils/                     # Configurations & helper functions
│       ├── config.py
│       └── config.yaml
│
├── .env                           # Environment variables
├── Makefile                       # Automate pipelines (install, data, train, inference)
└── README.md                      # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Telco-Customer-Churn-Prediction.git
   cd Telco-Customer-Churn-Prediction
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

Perfect 👍 thanks for clarifying! That means we should mention **two separate requirements files** in the README:

* One for running **Jupyter notebooks** in the `EDA/` folder.
* Another for running the **Python scripts & pipelines** in the `Python_Script/` folder.

Here’s the corrected part of the `README.md` (only the **Installation** section needs updating):

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Telco-Customer-Churn-Prediction.git
   cd Telco-Customer-Churn-Prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   * For **EDA notebooks**:

     ```bash
     pip install -r EDA/requirements.txt
     ```

   * For **Python scripts & pipelines**:

     ```bash
     pip install -r Python_Script/requirements.txt
     ```


---

## 🛠️ Usage

### Run Pipelines with Makefile

```bash
make install             # Set up environment and dependencies
make data-pipeline       # Run data preprocessing
make train-pipeline      # Train models
make streaming-inference # Run inference on sample JSON
```

### Direct Execution

```bash
python Python_Script/pipelines/data_pipeline.py
python Python_Script/pipelines/training_pipeline.py
python Python_Script/pipelines/streaming_inference_pipeline.py
```

---

## 📈 Future Work

* Integration with **MLflow & ZenML** for experiment tracking
* Orchestration with **Airflow**
* Real-time data streaming with **Kafka**
* Deployment on **AWS with REST API**
* Monitoring for production-readiness

---

## 📜 License

This project is licensed under the MIT License.

---



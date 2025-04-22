# apache_spark

🧠 Alzheimer's Prediction using Apache Spark
This project aims to predict Alzheimer's disease using clinical data from the OASIS (Open Access Series of Imaging Studies) dataset. The machine learning pipeline is implemented in PySpark using the Random Forest Classifier, making it scalable and suitable for large datasets.

📂 Project Structure
python
Copy
Edit
├── alzheimers_prediction.py                # Main PySpark script for model training
├── oasis_cross-sectional.csv              # Input dataset used for training
├── classification-models-for-dementia...  # Jupyter notebook for classification comparisons
├── archive.zip                             # Original dataset archive
├── oasis_longitudinal.csv                 # Additional dataset (not used in current script)
└── alzheimers_rf_model/                   # Saved Spark ML model
🧪 Dataset Description
Source: OASIS-1 Cross-Sectional Dataset

Target Variable: CDR (Clinical Dementia Rating)

Features Used:

Age, Education (Educ), Socioeconomic Status (SES)

Mini-Mental State Examination Score (MMSE)

Estimated Total Intracranial Volume (eTIV)

Normalized Whole Brain Volume (nWBV)

Atlas Scaling Factor (ASF)

Gender (M/F)

⚙️ Requirements
Python 3.x

Apache Spark & PySpark

Java 11+ (Ensure JAVA_HOME is set correctly)

Dataset file oasis_cross-sectional.csv

🚀 How to Run
Install PySpark:

bash
Copy
Edit
pip install pyspark
Set Environment Variables (Windows Example):

bash
Copy
Edit
set JAVA_HOME="C:\Program Files\Java\jdk-11.0.26+4"
set HADOOP_HOME="C:\winutils"
Run the Script:

bash
Copy
Edit
python alzheimers_prediction.py
Output:

Model accuracy printed in the terminal

Model saved to alzheimers_rf_model/

Predictions saved to alzheimers_predictions/

📊 Evaluation
Model: Random Forest Classifier (100 trees)

Metric: Accuracy (Multiclass Classification)

Label Mapping: CDR values are preprocessed to simplify into classes:

0.0 → 0 (No Dementia)

0.5 → 0 (Mild Dementia grouped)

1.0+ → 1 (Probable Dementia)

📘 Notes
All missing values are dropped during preprocessing.

Gender column is converted to numeric using StringIndexer.

The notebook classification-models-for-dementia... can be used to compare other classifiers.

📌 Future Work
Extend to use oasis_longitudinal.csv for temporal predictions

Use multiclass classification for better granularity

Add visualization for feature importance and confusion matrix

🧑‍💻 Author
Developed as part of a B.Tech project on early diagnosis of dementia using scalable ML tools.

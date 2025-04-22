import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor, when
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Set Hadoop Home Path (Windows Fix)
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-11.0.26+4"
os.environ["HADOOP_HOME"] = "C:\\winutils"
os.environ["PATH"] += ";C:\\winutils\\bin"

# Start Spark Session
spark = SparkSession.builder \
    .appName("AlzheimerPrediction") \
    .config("spark.sql.repl.eagerEval.enabled", True) \
    .config("spark.hadoop.security.authentication", "nosasl") \
    .config("spark.authenticate", "false") \
    .getOrCreate()

# Load Dataset
file_path = "D:\\SEM6\\AI Hardware and tools workshop\\project\\unit4\\Major\\alzheimers apache spark\\alzheimers apache spark\\oasis_cross-sectional.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check for correct target column
if "CDR" not in df.columns:
    raise ValueError("Target column 'CDR' (Clinical Dementia Rating) not found in dataset")

# Drop unnecessary columns
columns_to_drop = ["ID", "Hand", "Delay"]
df = df.drop(*[col for col in columns_to_drop if col in df.columns])

# Handle missing values
df = df.na.drop()

# Convert categorical columns to numeric
if "M/F" in df.columns:
    indexer = StringIndexer(inputCol="M/F", outputCol="M/F_index", handleInvalid="keep").fit(df)
    df = indexer.transform(df)

# Feature Engineering
feature_cols = [col for col in ["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "M/F_index"] if col in df.columns]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Convert label column to integer (Fixing Label Issue)
df = df.withColumn("label", col("CDR").cast(DoubleType()))
df = df.withColumn("label", floor(col("label")))  # Convert float labels to integers
df = df.withColumn("label", when(col("label") == 0.5, 0).otherwise(col("label").cast("int")))

# Check dataset size
if df.count() < 10:
    raise ValueError("Dataset has too few samples for model training.")

# Check class distribution
df.groupBy("label").count().show()

# Split Data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train Random Forest Model
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
model = rf.fit(train_df)  # ✅ No more label issue!

# Make Predictions
predictions = model.transform(test_df)

# Evaluate Model
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Save Model
model.write().overwrite().save("alzheimers_rf_model")

# Save Predictions
predictions.select("label", "prediction").write.csv("alzheimers_predictions", header=True, mode="overwrite")

# Stop Spark Session
spark.stop()
print("✅ Alzheimer's Prediction Model Training Completed!")

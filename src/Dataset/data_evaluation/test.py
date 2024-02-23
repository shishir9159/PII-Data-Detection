import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

df = pd.read_csv('src\Dataset\pii_detection_removal_from_educational_dataset_with_4434_essays.csv')

unique_value_in_percent = {}

for col in df.columns:
    unique_values = df[col].nunique()
    percentage = (unique_values / df[col].count()) * 100
    percentage_unique[col] = percentage


for col, percentage in percentage_unique.items():
    print(f'The column "{col}" has {percentage:.2f}% unique values.')

unique_values = df['username'].nunique()

print(unique_values)
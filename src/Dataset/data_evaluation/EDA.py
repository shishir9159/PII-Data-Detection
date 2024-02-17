import pandas as pd

# Load the dataset
df = pd.read_csv('Dataset\pii_detection_removal_from_educational_dataset_with_4434_essays.csv')

# Calculate length of each entry in the desired column
df['length'] = df['document'].apply(len)

# Get length distribution
length_distribution = df['length'].value_counts().sort_index()

print(length_distribution)

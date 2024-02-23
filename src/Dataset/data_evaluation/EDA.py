import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

df = pd.read_csv('src\Dataset\pii_detection_removal_from_educational_dataset_with_4434_essays.csv')
df['text_length'] = df['text'].apply(len)

length_distribution = df['text_length'].value_counts().sort_index()

print(length_distribution)

# Plot length distribution
plt.figure(figsize=(10, 6))
# plt.bar(length_distribution.index, length_distribution.values)
# plt.hist(df['text_length'], bins=50, alpha=0.5, color='g')
plt.boxplot(df['text_length'], vert=False)

plt.xlabel('Length of Text')
plt.ylabel('Frequency')
plt.title('Text Length Distribution')
plt.grid(True)
plt.show()


#df['num_tokens'] = df['tokens'].apply(word_tokenize)
df['num_tokens'] = df['tokens'].apply(len)

# Get token distribution
token_distribution = df['num_tokens'].value_counts().sort_index()

# Plot token distribution
plt.figure(figsize=(10, 6))
# plt.bar(token_distribution.index, token_distribution.values)
# plt.hist(df['num_tokens'], bins=50, alpha=0.5, color='g')
plt.boxplot(df['num_tokens'], vert=False)

plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.title('Token Distribution')
plt.grid(True)
plt.show()
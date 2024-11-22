import pandas as pd

# Load the CSV file into a DataFrame
file_path = "data/emotion_dataset_testing.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Inspect the DataFrame
print(df.head())
print(df['Emotion'].unique())

# Filter out rows where 'Emotion' equals integer 5
df = df[df['Emotion'].apply(lambda x: str(x) != '5')]

# Save the updated DataFrame back to the CSV file
df.to_csv(file_path, index=False)

print("Rows with 'Emotion' value 5 have been removed.")

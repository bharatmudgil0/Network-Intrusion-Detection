import pandas as pd

df = pd.read_csv(
    "dataset/processed/model_dataset.csv"
)

# Select 100 records
sample = df.sample(
    n=100,
    random_state=42
)

# Remove labels because the dashboard predicts them
sample = sample.drop(
    columns=["Label"]
)

sample.to_csv(
    "dataset/processed/test_sample.csv",
    index=False
)

print("Test sample created successfully!")
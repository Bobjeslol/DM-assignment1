import pandas as pd
import numpy as np

'''To check the train and test data before training the RNN-classification model'''

df_train = pd.read_csv("train_RNN.csv")
df_test = pd.read_csv("test_RNN.csv")

print("Columns:", df_train.columns.tolist())
assert all(col in df_train.columns for col in ["sequence_id", "timestep", "var_id", "value", "delta_t", "target"]), "Missing columns"

#train/test shapes
train_seq_lengths = df_train.groupby("sequence_id")["timestep"].nunique()
test_seq_lengths = df_test.groupby("sequence_id")["timestep"].nunique()
print(f"Train sequence length (unique): {train_seq_lengths.unique()}")
print(f"Test sequence length (unique): {test_seq_lengths.unique()}")
assert train_seq_lengths.nunique() == 1, "Inconsistent sequence lengths in train"
assert test_seq_lengths.nunique() == 1, "Inconsistent sequence lengths in test"

#normalization check
print("\nValue range by var_id (train):")
for var_id in sorted(df_train["var_id"].unique()):
    if var_id == -1:
        continue
    values = df_train[df_train["var_id"] == var_id]["value"]
    print(f"var_id={int(var_id)} — min={values.min():.4f}, max={values.max():.4f}")

#'Mood' distribution
print("\nTarget mood values (train):")
print(df_train.groupby("sequence_id")["target"].first().describe())
print("Unique mood values:", df_train["target"].unique())

#missing values check
for df_name, df in [("train", df_train), ("test", df_test)]:
    if df.isnull().any().any():
        print(f"{df_name} contains missing values")
    else:
        print(f"{df_name} is clean.")

#overlap train/test data
train_ids = set(df_train["sequence_id"].unique())
test_ids = set(df_test["sequence_id"].unique())
overlap = train_ids & test_ids
print(f"\nOverlapping sequence IDs in train/test: {len(overlap)}")
assert len(overlap) == 0, "There areoverlapping sequence IDs"

#delta_t checks
assert (df_train["delta_t"] >= 0).all(), "Negative delta_t in train"
assert (df_test["delta_t"] >= 0).all(), "Negative delta_t in test"
print(" delta_t values are all non-negative.")
values_delta = df_train["delta_t"].values
print(f"delta_t — min={values_delta.min():.4f}, max={values_delta.max():.4f}")

print("\n All checks passed")

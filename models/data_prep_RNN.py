import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

#mapping var_id to variable name
var_id_to_name = {
    0: 'activity',
    1: 'appCat.builtin',
    2: 'appCat.communication',
    3: 'appCat.entertainment',
    4: 'appCat.finance',
    5: 'appCat.game',
    6: 'appCat.office',
    7: 'appCat.other',
    8: 'appCat.social',
    9: 'appCat.travel',
    10: 'appCat.unknown',
    11: 'appCat.utilities',
    12: 'appCat.weather',
    13: 'call',
    14: 'circumplex.arousal',
    15: 'circumplex.valence',
    16: 'mood',
    17: 'screen',
    18: 'sms'
}
binary_vars = {"activity", "call", "sms"}

def prepare_event_sequences(df, sequence_length=7, step_size=3):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    variable_index = {var: i for i, var in enumerate(df["variable"].unique())}
    df["var_id"] = df["variable"].map(variable_index)

    sequences, targets, meta = [], [], []

    for _, group in df.groupby("id"):
        #keep track of time gaps between mood measurements and skips if longer than 2 days
        group = group.sort_values("time").reset_index(drop=True)
        mood_df = group[group["variable"] == "mood"].copy()
        mood_df["mood_gap"] = mood_df["time"].diff().dt.days.fillna(0)
        gap_indices = mood_df.index[mood_df["mood_gap"] > 2].tolist()
        split_points = [0] + gap_indices + [len(group)]
        chunks = [group.iloc[split_points[i]:split_points[i + 1]] for i in range(len(split_points) - 1)]

        #skip if less than 5 mood measurements
        for chunk in chunks:
            chunk = chunk.sort_values("time").reset_index(drop=True)
            if len(chunk) < 5:
                continue

            prev_time = None
            events, times = [], []
            for _, row in chunk.iterrows():
                var_id = row["var_id"]
                value = row["value"]
                ts = row["time"]
                delta_t = (ts - prev_time).total_seconds() if prev_time else 0
                prev_time = ts
                events.append([var_id, value, delta_t])
                times.append(ts)

            moods = [(i, row.value) for i, row in enumerate(chunk.itertuples()) if row.variable == "mood"]

            #create sequences of events
            for i in range(0, len(events) - sequence_length, step_size):
                end_time = times[i + sequence_length - 1]
                target_time = end_time + pd.Timedelta(days=1)
                future_moods = [mood_val for idx, mood_val in moods if times[idx] >= target_time]
                if future_moods:
                    sequences.append(np.array(events[i:i + sequence_length]))
                    targets.append(future_moods[0])
                    meta.append((group["id"].iloc[0], end_time))

    return np.array(sequences), np.array(targets), meta, variable_index

def normalize_per_variable(sequences, var_id_to_name, binary_vars):
    #normalize each variable separately due to different scales
    scalers = {}
    unique_var_ids = np.unique(sequences[:, :, 0])
    for var_id in unique_var_ids:
        if var_id == -1:
            continue
        var_name = var_id_to_name[int(var_id)]
        if var_name in binary_vars:
            scalers[int(var_id)] = None
            continue
        mask = sequences[:, :, 0] == var_id
        values = sequences[mask][:, 1].reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(values)
        sequences[mask, 1] = scaler.transform(values).flatten()
        scalers[int(var_id)] = scaler

    delta_scaler = MinMaxScaler()
    delta_scaler.fit(sequences[:, :, 2].reshape(-1, 1))
    sequences[:, :, 2] = delta_scaler.transform(sequences[:, :, 2].reshape(-1, 1)).reshape(sequences.shape[0], sequences.shape[1])
    scalers["delta_t"] = delta_scaler
    return sequences, scalers

def to_dataframe(X, y, start_id=0):
    flat = []
    for i, (seq, target) in enumerate(zip(X, y)):
        seq_id = start_id + i
        for t, (var_id, val, dt) in enumerate(seq):
            flat.append([seq_id, t, var_id, val, dt, target])
    return pd.DataFrame(flat, columns=["sequence_id", "timestep", "var_id", "value", "delta_t", "target"])

def run_all(input_csv="imputed_data.csv", sequence_length=7, step_size=3):
    df = pd.read_csv(input_csv)
    X, y, meta, _ = prepare_event_sequences(df, sequence_length=sequence_length, step_size=step_size)
    X_scaled, scalers = normalize_per_variable(X, var_id_to_name, binary_vars)

    meta_df = pd.DataFrame(meta, columns=["id", "time"])
    meta_df["index"] = np.arange(len(meta_df))
    meta_df = meta_df.sort_values(["id", "time"]).reset_index(drop=True)

    #split into train and test set temporal
    train_idx, test_idx = [], []
    for _, user_df in meta_df.groupby("id"):
        n = len(user_df)
        split = int(n * 0.8)
        train_idx.extend(user_df.iloc[:split]["index"].tolist())
        test_idx.extend(user_df.iloc[split:]["index"].tolist())

    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_test, y_test = X_scaled[test_idx], y[test_idx]

    joblib.dump(scalers, "value_scalers_per_variable.pkl")

    df_train = to_dataframe(X_train, y_train, start_id=0)
    df_test = to_dataframe(X_test, y_test, start_id=len(X_train))
    df_train.to_csv("train_RNN.csv", index=False)
    df_test.to_csv("test_RNN.csv", index=False)

    print(f"Saved train_RNN.csv ({len(X_train)} sequences)")
    print(f"Saved test_RNN.csv ({len(X_test)} sequences)")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_all("imputed_data.csv")

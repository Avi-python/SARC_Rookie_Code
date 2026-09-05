# %%
import pyarrow.parquet as pq
import pandas as pd
import soundfile as sf
import io

# %%
path = "/nfs/DGX2/Workspace/avi/processed_data_independent3/tsu/corpus=formosa/split=train/language=tsu_Latn/3_f672e0ce81404625af4c7c3418f00e15_000000_000000-0.parquet"
# 1. Just the schema (column names + types) — fast, no data loaded
schema = pq.read_schema(path)
print(schema)

# %%
df = pd.read_parquet(path)
print(df.columns.tolist())
print(df.head(3))
print(df.shape)  # (num_rows, num_cols)

# %%

row = df["audio_bytes"].iloc[0]
raw = row.tobytes()
audio_array, sample_rate = sf.read(io.BytesIO(raw))
print(audio_array.shape, sample_rate, audio_array.dtype)
print(audio_array)

# %%
df['audio_bytes'].shape

# %%
row = df['audio_bytes'].iloc[0]
print(row.shape)
print(row.dtype)

# %%
index = 1
row = df["audio_bytes"].iloc[index]
print(type(row))
print(row.shape if hasattr(row, "shape") else len(row))
print(row.dtype if hasattr(row, "dtype") else "no dtype - probably raw bytes")

print(df["audio_size"].iloc[index])   # compare vs shape/length above
print(df["text"].iloc[index])         # see what the transcription looks like

# %%
with open("test.wav", "wb") as f:
    f.write(row.tobytes())  # or bytes(row) depending on dtype

# %%

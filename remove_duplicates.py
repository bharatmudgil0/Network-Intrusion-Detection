
import os
import sqlite3
import numpy as np
import pandas as pd

INPUT_FILE = "dataset/processed/cleaned_cicids2017.csv"
OUTPUT_FILE = "dataset/processed/final_cicids2017.csv"
DB_FILE = "dataset/processed/duplicate_hashes.sqlite"

# Start fresh each time
for path in [OUTPUT_FILE, DB_FILE]:
    if os.path.exists(path):
        os.remove(path)

# SQLite database stores fingerprints of previously seen rows
conn = sqlite3.connect(DB_FILE)

conn.execute("""
    CREATE TABLE seen (
        fingerprint INTEGER PRIMARY KEY
    )
""")

conn.execute("""
    CREATE TEMP TABLE current (
        fingerprint INTEGER PRIMARY KEY
    )
""")

total_rows = 0
unique_rows = 0
write_header = True

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=100000,
    low_memory=False
):

    total_rows += len(chunk)

    # Generate a fingerprint for every complete row
    hashes = pd.util.hash_pandas_object(
        chunk,
        index=False
    ).to_numpy(dtype=np.uint64)

    # SQLite INTEGER uses signed 64-bit integers
    hashes = hashes.view(np.int64)

    # Identify repeated fingerprints within this chunk
    first_occurrence = ~pd.Series(hashes).duplicated().to_numpy()

    unique_hashes = hashes[first_occurrence]

    # Clear fingerprints from the previous chunk
    conn.execute("DELETE FROM current")

    # Insert this chunk's unique fingerprints
    conn.executemany(
        "INSERT OR IGNORE INTO current VALUES (?)",
        [(int(h),) for h in unique_hashes]
    )

    # Find fingerprints not encountered in earlier chunks
    new_hashes = np.array([
        row[0] for row in conn.execute("""
            SELECT current.fingerprint
            FROM current
            LEFT JOIN seen
            ON current.fingerprint = seen.fingerprint
            WHERE seen.fingerprint IS NULL
        """).fetchall()
    ], dtype=np.int64)

    # Keep only the first occurrence of each new fingerprint
    keep = (
        first_occurrence &
        np.isin(hashes, new_hashes)
    )

    cleaned_chunk = chunk.loc[keep]

    # Save newly discovered fingerprints
    conn.executemany(
        "INSERT OR IGNORE INTO seen VALUES (?)",
        [(int(h),) for h in new_hashes]
    )

    conn.commit()

    # Write unique records to the output CSV
    cleaned_chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if write_header else "a",
        header=write_header,
        index=False
    )

    if len(cleaned_chunk) > 0:
        write_header = False

    unique_rows += len(cleaned_chunk)

    print(
        f"Processed {total_rows:,} records | "
        f"Unique records: {unique_rows:,}"
    )

conn.close()

print("\n" + "=" * 50)
print("DUPLICATE REMOVAL COMPLETED")
print("=" * 50)

print("Records before:", total_rows)
print("Unique records:", unique_rows)
print("Duplicate records:", total_rows - unique_rows)
print("Saved file:", OUTPUT_FILE)

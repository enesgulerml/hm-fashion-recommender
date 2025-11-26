import pandas as pd
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'transactions_train.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'transactions_optimized.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')

CHUNK_SIZE = 100000
START_DATE = '2020-08-01'


def process_large_data():
    # Folder control
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Folder created: {OUTPUT_DIR}")

    print(f"The process begins...")
    print(f"Source: {INPUT_FILE}")
    print(f"Target: {OUTPUT_FILE}")
    print(f"Filter: Data after {START_DATE}.")

    first_chunk = True
    total_rows = 0

    try:
        for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):

            chunk['t_dat'] = pd.to_datetime(chunk['t_dat'])

            filtered_chunk = chunk[chunk['t_dat'] >= START_DATE]

            if not filtered_chunk.empty:
                mode = 'w' if first_chunk else 'a'
                header = first_chunk

                filtered_chunk.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)

                total_rows += len(filtered_chunk)
                first_chunk = False
                print(f".{len(filtered_chunk)} rows processed and added.")

    except FileNotFoundError:
        print("ERROR: transactions_train.csv file not found in data/raw!")
        return

    print("-" * 30)
    print(f"DONE! Total {total_rows} rows filtered and saved.")
    print(f"New file location: {OUTPUT_FILE}")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    process_large_data()
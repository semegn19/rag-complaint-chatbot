import numpy as np
import faiss

def get_row_by_index(parquet_file, target_index):
    """
    Efficiently fetch a row by scanning batches.
    (works without loading full dataset into RAM)
    """
    current = 0

    for batch in parquet_file.iter_batches(batch_size=5000):
        df = batch.to_pandas()

        if current + len(df) > target_index:
            return df.iloc[target_index - current]

        current += len(df)

    return None


def retrieve(question,
             embedding_model,
             index,
             row_mapping,
             parquet,
             k=5):

    q = embedding_model.encode(question)
    q = np.array([q]).astype("float32")

    faiss.normalize_L2(q)

    scores, ids = index.search(q, k)

    results = []

    for idx in ids[0]:

        row_number = row_mapping[idx]

        row = get_row_by_index(parquet, row_number)

        results.append(row)

    return results
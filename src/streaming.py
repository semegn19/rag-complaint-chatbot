import numpy as np
import pyarrow.parquet as pq
import faiss
import pickle

def build_streaming_index(parquet_path,
                          index_path="complaints.index",
                          mapping_path="row_mapping.pkl",
                          batch_size=5000):

    parquet = pq.ParquetFile(parquet_path)

    index = None

    row_mapping = {}

    global_row = 0

    for batch in parquet.iter_batches(batch_size=batch_size):

        df = batch.to_pandas()

        vectors = np.vstack(df["embedding"].values).astype("float32")

        faiss.normalize_L2(vectors)

        if index is None:
            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)

        index.add(vectors)

        for i in range(len(df)):
            row_mapping[index.ntotal - len(df) + i] = global_row
            global_row += 1

        print(f"Indexed {index.ntotal:,} vectors")

    faiss.write_index(index, index_path)

    with open(mapping_path, "wb") as f:
        pickle.dump(row_mapping, f)

    print("Finished.")
import numpy as np

class SHAPMapper:
    """
    Map topic IDs -> design semantic factor vectors (S_k in the paper).
    Minimal reproducible mapping:
    - One-hot topic vector (K dims)
    """

    def __init__(self, num_factors=12):
        self.K = int(num_factors)

    def map_topics_to_factors(self, topic_ids):
        factors = []
        for k in topic_ids:
            v = np.zeros(self.K, dtype=np.float32)
            v[int(k) % self.K] = 1.0
            factors.append(v)
        return np.stack(factors, axis=0)

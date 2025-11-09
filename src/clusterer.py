from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict


def cluster_questions(questions, embeddings, eps=0.35, min_samples=1):
    # normalize for cosine
    X = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
    labels = clustering.labels_
    clusters = defaultdict(list)
    for i, lab in enumerate(labels):
        clusters[lab].append(questions[i])
    # build cluster_info sorted by size
    cluster_info = []
    for lab, members in clusters.items():
        cluster_info.append((lab, {'count': len(members), 'examples': members[:5]}))
    cluster_info.sort(key=lambda x: -x[1]['count'])
    return labels, cluster_info
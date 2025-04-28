import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def find_optimal_eps(X_scaled, min_samples=3):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    distances = np.sort(distances[:, min_samples-1])

    kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
    optimal_eps = distances[kneedle.elbow]

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-th nearest neighbor distance')
    plt.title('Elbow Method for Optimal eps')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_eps

def find_optimal_eps_varying_min_samples(X_scaled, min_samples_list=[3, 4, 5, 6]):
    results = []

    for min_samples in min_samples_list:
        neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances, _ = neighbors.kneighbors(X_scaled)
        distances = np.sort(distances[:, min_samples-1])

        try:
            kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
            elbow_index = kneedle.elbow
            if elbow_index is not None:
                optimal_eps = distances[elbow_index]

                # Optimal eps ile DBSCAN uygulayıp kaç küme oluşuyor bakalım
                dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                results.append({
                    "min_samples": min_samples,
                    "optimal_eps": float(optimal_eps),
                    "n_clusters": n_clusters
                })
            else:
                results.append({
                    "min_samples": min_samples,
                    "optimal_eps": None,
                    "n_clusters": 0
                })
        except Exception as e:
            results.append({
                "min_samples": min_samples,
                "optimal_eps": None,
                "n_clusters": 0,
                "error": str(e)
            })

    return results

def perform_dbscan(df, feature_columns, min_samples=3):
    # X = df[['avg_price', 'sales_frequency', 'avg_quantity_per_order', 'unique_customers']]
    X = df[feature_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    optimal_eps = find_optimal_eps(X_scaled, min_samples)

    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    df['cluster'] = dbscan.fit_predict(X_scaled)

    return df
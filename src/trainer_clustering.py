from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def get_pca_components(data, variance_limit=0.9999):
    """
    Return the number of components in PCA for a limit
    in variance explained to be reached
    """
    
    # Train a PCA and get explained variance
    pca = PCA().fit(data)
    explained_variance = pca.explained_variance_ratio_
    
    # Optimize number of components
    n, total_variance = 0, 0
    while total_variance < variance_limit:
        total_variance = sum(explained_variance[:n])
        n +=1
    
    return n-1

def plot_cluster(X):

    x = "pca_1"
    y = "pca_2"
    
    # Black removed and is used for noise instead.
    unique_labels = set(X.labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(20,10))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        X['class_member_mask'] = X['labels'] == k

        xy = X.loc[(X.class_member_mask == True) & (X.core_samples == True)]
        plt.plot(xy[[x]], xy[[y]], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X.loc[(X.class_member_mask == True) & (X.core_samples == False)]
        plt.plot(xy[[x]], xy[[y]], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.xlabel(x)
    plt.ylabel(y)
    
    
def train_DBSCAN(data, pca_components=2, eps=0.3, min_samples=3):
    
    # PCA
    pca = PCA(pca_components).fit(data)
    explained_variance = pca.explained_variance_ratio_
    pca_data = pca.transform(data)
    for n in range(pca_components):
        data["pca_" + str(n+1)] = pca_data[:,n]
    
    # Normalizamos los datos
    X = StandardScaler().fit_transform(pca_data)

    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    # Extract labels & core sample
    labels = db.labels_
    data['labels'] = labels
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    data['core_samples'] = core_samples_mask

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Metrics
    metrics = {"n_features": X.shape[1],
               "epsilon": eps,
               "min_samples": min_samples,
               "n_clusters": n_clusters,
               "n_noise": n_noise,
               "silhouette": silhouette_score(X, labels),
               
               "explained_pca_variance":sum(explained_variance)}

    print("Metrics:")
    print(metrics)
    
    return metrics, data

def search_best(data):
    full_metrics = []
    for pca_components in range(2,5):
        for eps in np.linspace(0.1,1,10):
            for min_samples in range(5):
                try:
                    metrics, results = train_DBSCAN(data, pca_components, eps, min_samples)
                    full_metrics.append(metrics)
                    plot_cluster(results)
                    plt.savefig("reports/clustering/pca2_{}feat_{}eps_{}minsamples.png"\
                        .format(pca_components, eps, min_samples))
                except:
                    print("")
                    print("[ERROR] Problem training model with pca_components: {}, eps {} and min samples {}"\
                         .format(pca_components, eps, min_samples))
                    print("")
                    pass
    all_metrics = pd.DataFrame(full_metrics).sort_values('silhouette', ascending=False)
    sns.pairplot(all_metrics).savefig("reports/clustering/metrics_pairplot.png")
    
    return all_metrics
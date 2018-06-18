import collections

from sklearn import cluster
import DataPreprocessor


def clustering(data, activities, n_cluster):
    kmeans_fit = cluster.KMeans(n_clusters=n_cluster).fit(data)
    cluster_to_activities = collections.defaultdict(list)

    cluster_labels = kmeans_fit.labels_
    for i in range(len(data)):
        index_cluster = cluster_labels[i]
        cluster_to_activities[index_cluster].append(activities[i])

    for i in range(n_cluster):
        print("\n\nActivities in cluster " + str(i) + ", Count: " + str(len(cluster_to_activities[i])))
        for activity in cluster_to_activities[i][0:20]:
            print(activity['id'], activity['title'].strip().replace('\n', ''),
                 activity['content'].strip().replace('\n', ''))


if __name__ == '__main__':
    preprocessor = DataPreprocessor.DataPreprocessor()
    data, _ = preprocessor.get_words_embedding_preprocessing()
    clustering(data, preprocessor.activities, 20)
def clustering():
    data, labels = self.get_all_normalized_data_labels()
    kmeans_fit = cluster.KMeans(n_clusters=n_cluster).fit(data)
    activities_in_clusters = [[] for i in range(n_cluster)]
    wordset_in_clusters = [set() for i in range(n_cluster)]

    cluster_labels = kmeans_fit.labels_
    for i in range(len(data)):
        index_cluster = cluster_labels[i]
        activities_in_clusters[index_cluster].append(self.activities[i])

    for i in range(n_cluster):
        wordset_in_clusters[i] = None
        for activity in activities_in_clusters[i]:
            words = set(self.cut_activity_to_words(activity))
            if not wordset_in_clusters[i]:
                wordset_in_clusters[i] = words
            else:
                wordset_in_clusters[i] &= words

    if show_result:
        for i in range(n_cluster):
            print("\n\nActivities in cluster " + str(i) + ", Count: " + str(len(activities_in_clusters[i])))
            print("Common words: ", wordset_in_clusters[i])
            for activity in activities_in_clusters[i][0:n_showing_activities_each_cluster]:
                print(activity['id'], activity['title'].strip().replace('\n', ''),
                      activity['content'].strip().replace('\n', ''))


if __name__ == '__main__':
    clustering()
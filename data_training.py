

class ActivityCategorizationTrainer:
    def __init__(self, data, labels):
        count = len(data)
        self.train_data = data[:count*0.7, :]
        self.train_labels = labels[:count*0.7, :]
        self.test_data = data[count*0.7:, :]
        self.test_labels = labels[count*0.7, :]

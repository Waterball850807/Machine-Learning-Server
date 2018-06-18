from keras.callbacks import Callback


class WeightsSaver(Callback):
    def __init__(self, model, N, filename):
        self.model = model
        self.N = N
        self.filename = filename
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = self.filename
            self.model.save_weights(name)
        self.batch += 1
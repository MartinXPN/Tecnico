from betlearn import BetLearn


def main():
    # Instructions to load the model later:
    # import tensorfloas as tf
    # from tensorflow.keras.models import load_model
    # load_model('path/to/experiments/<DATE>/models/best.h5py', custom_objects={'tf': tf}, compile=False).summary()

    btl = BetLearn(min_nodes=50, max_nodes=100, nb_train_graphs=1000, nb_valid_graphs=100, graphs_per_batch=16, nb_batches=50,
                   graph_type='powerlaw', optimizer='adam', aggregation='max', combine='gru')
    btl.train(epochs=10000)


if __name__ == "__main__":
    main()

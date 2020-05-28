from betlearn import BetLearn


def main():
    btl = BetLearn(min_nodes=100, max_nodes=200, nb_train_graphs=1000, nb_valid_graphs=100, graphs_per_batch=16, nb_batches=500,
                   graph_type='powerlaw', optimizer='adam', aggregation='lstm', combine='gru')
    btl.train(epochs=10000)


if __name__ == "__main__":
    main()

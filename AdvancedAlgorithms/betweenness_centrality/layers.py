import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, GRU, Dense


@tf.keras.utils.register_keras_serializable(package='drbc', name='DrBCRNN')
class DrBCRNN(Layer):
    def __init__(self, units=128, repetitions=5, combine='gru', return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.repetitions = repetitions
        self.combine_method = combine
        self.return_sequences = return_sequences

        combine = combine.strip().lower()
        if combine == 'gru':
            self.combine = GRU(units=units, return_sequences=False)
        else:
            raise ValueError(f'Combine method `{combine}` is not implemented yet!')

        self.node_linear = Dense(self.units)

    def call(self, inputs, **kwargs):
        n2n, message = inputs
        states = [message]
        for rep in range(self.repetitions):
            n2n_pool = tf.sparse.sparse_dense_matmul(n2n, states[rep])
            # print(n2n_pool)
            node_representations = self.node_linear(n2n_pool)
            combined = self.combine(tf.expand_dims(node_representations, 1))
            res = K.l2_normalize(combined, axis=1)
            states.append(res)

        if not self.return_sequences:
            return states[-1]

        # B x embeding_dim x repetitions
        target_shape = [dim if dim is not None else -1 for dim in K.int_shape(message)]
        target_shape.append(self.repetitions)
        out = tf.concat(states[1:], axis=-1)
        out = tf.reshape(out, shape=target_shape)
        return out

    def get_config(self):
        return {
            'units': self.units,
            'repetitions': self.repetitions,
            'combine': self.combine_method,
            'return_sequences': self.return_sequences,
        }

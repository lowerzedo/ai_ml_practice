# Import TensorFlow
import tensorflow as tf

# Import numpy
import numpy as np

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)


model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])


model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())

model.fit(xs, ys, epochs=500)

cloud_logger.info(str(model.predict([10.0])))
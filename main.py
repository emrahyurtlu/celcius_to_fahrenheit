import logging

import numpy as np
import tensorflow as tf

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

X_train = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
print(X_train)
Y_train = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
print(Y_train)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([layer])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(X_train, Y_train, epochs=500, verbose=False)

param = float(input("Enter new value:"))

prediction = model.predict([param])
print(f"Your prediction result is: {prediction}")

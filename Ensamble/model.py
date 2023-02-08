import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model1 = tf.keras.models.load_model(os.path.join(path, 'V2B2_1'))
        self.model2 = tf.keras.models.load_model(os.path.join(path, 'B5'))
        self.model3 = tf.keras.models.load_model(os.path.join(path, 'ConvNext'))
        self.model4 = tf.keras.models.load_model(os.path.join(path, 'V2B2_2'))
        self.model5 = tf.keras.models.load_model(os.path.join(path, 'Xception'))

    def predict(self, X):
        
        # Insert your preprocessing here

        out1 = self.model1.predict(X)
        out2 = self.model2.predict(X)
        out3 = self.model3.predict(tf.image.resize(X, (256,256)))
        out4 = self.model4.predict(tf.image.resize(X, (256, 256)))
        out5 = self.model5.predict(tf.image.resize(X, (256, 256)))
        out = (out1 + out2 + out3 + out4 + out5) / 5
        out = tf.argmax(out, axis=-1)

        return out
import tensorflow as tf


class FruitNet:
    def __init__(self, model_config_path):
        self.model = tf.keras.models.load_model(model_config_path)

    def predict(self, img_file):
        labels = ['Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites','Tomato_Target_Spot','Tomato_Yellow_Leaf_Curl_Virus','Tomato_mosaic_virus','Tomato_healthy']

        img_tensor = tf.image.decode_image(img_file)
        img_resized = tf.image.resize(img_tensor, [256, 256])
        img_final = tf.expand_dims(img_resized, 0)

        y_probs = self.model.predict(img_final[:,:,:,:3])
        y_label = y_probs.argmax(axis=-1)

        label = labels[y_label[0]]
        percentage = format(y_probs.max() * 100, '.1f')

        return label, percentage

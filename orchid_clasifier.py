import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
class OrchidClassifierTensorFlow:
    def __init__(self):
        self.train_datagen = ImageDataGenerator(rescale=1./255)
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
            'train',
            target_size=(150, 150),
            batch_size=20,
            class_mode='categorical')

        self.validation_generator = self.test_datagen.flow_from_directory(
            'test',
            target_size=(150, 150),
            batch_size=20,
            class_mode='categorical')

        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(52, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train(self, epochs=10):
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=100,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=50
        )

    def save_model(self, path):
        self.model.save(path)

class OrchidPredictorTensorFlow:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['anoectochilus burmanicus rolfe', 'bulbophyllum auricomum lindl', 'bulbophyllum dayanum rchb', 'bulbophyllum lasiochilum par. & rchb', 'bulbophyllum limbatum', 'bulbophyllum longissimum (ridl.) ridl', 'bulbophyllum medusae (lindl.) rchb', 'bulbophyllum patens king ex hk.f.', 'bulbophyllum rufuslabram', 'bulbophyllum siamensis rchb', 'calenthe rubens', 'chiloschista parishii seidenf.', 'chiloschista viridiflora seidenf.', 'cymbidium aloifolium (l.) sw.', 'dendrobium chrysotoxum lindl', 'dendrobium farmeri paxt.', 'dendrobium fimbriatum hook', 'dendrobium lindleyi steud', 'dendrobium pulchellum roxb', 'dendrobium pulchellum', 'dendrobium secundum bl-lindl', 'dendrobium senile par. & rchb.f.', 'dendrobium signatum rchb. f', 'dendrobium thyrsiflorum rchb. f.', 'dendrobium tortile lindl', 'dendrobium tortile', 'hygrochillus parishii var. marrioftiana (rchb.f.)', 'paphiopedilum bellatulum', 'paphiopedilum callosum', 'paphiopedilum charlesworthii', 'paphiopedilum concolor', 'paphiopedilum exul', 'paphiopedilum godefroyae', 'paphiopedilum gratrixianum', 'paphiopedilum henryanum', 'paphiopedilum intanon-villosum', 'paphiopedilum niveum (rchb.f.) stein', 'paphiopedilum parishii', 'paphiopedilum spicerianum', 'paphiopedilum sukhakulii', 'pelatantheria bicuspidata (rolfe ex downie) tang & wang', 'pelatantheria insectiflora (rchb.f.) ridl.', "phaius tankervilleae (banks ex i' heritier) blume", 'phalaenopsis cornucervi (breda) bl. & rchb.f.', 'rhynchostylis gigantea (lindl.) ridl.', 'trichoglottis orchideae (koern) garay.', 'bulbophyllum auratum Lindl.', 'bulbophyllum morphologorum Krzl.', 'dendrobium cumulatum Lindl.', 'maxiralia tenui folia', 'paphiopedilum vejvarutianum O. Gruss & Roellke', 'oncidium goldiana']  # Define your class names here

    def predict(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        predictions = self.model.predict(img_array)
        predicted_class = self.class_names[np.argmax(predictions[0])]
        return predicted_class
if __name__ == '__main__':
    #orchids=OrchidClassifierTensorFlow()
    #orchids.train()
    #orchids.save_model("orchids2.h5")
    pred=OrchidPredictorTensorFlow("orchids2.h5")
    print(pred.predict("T_Rhynchostylis-gigantea-Lindl-Ridl-ช้างกระ_79.JPEG"))

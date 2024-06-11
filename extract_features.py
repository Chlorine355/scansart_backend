import os
import keras
import keras.utils as image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from scipy.spatial import distance
import psycopg2


conn = psycopg2.connect(database="scanart", user="postgres", password="*", host="127.0.0.1", port='5432')
cursor = conn.cursor()
print("Connection Successful to PostgreSQL")
model = keras.applications.VGG16(weights='imagenet', include_top=True)
# print(model.summary())


def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def feat_to_postgres_double_array(fs):    return '{' + str(list(fs))[1:-1] + '}'


feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)


cursor.execute("SELECT path from images where features is null")
images = [x[0] for x in cursor.fetchall()]


print("keeping %d images to analyze" % len(images))
tic = time.time()
features = []
for i, image_path in enumerate(images):
    if i % 20 == 0:
        toc = time.time()
        elap = toc-tic
        print("analyzing image %d / %d. Time: %4.4f seconds." % (i, len(images), elap))
    try:
        img, x = load_image(image_path)
        feat = feat_extractor.predict(x)[0]
        features.append((image_path, feat))
    except Exception as e:
        print(e)

print('finished extracting features for %d images' % len(images))
for feature in features:
    cursor.execute(f"UPDATE images set features='{feat_to_postgres_double_array(feature[1])}' where path='{feature[0]}'")
conn.commit()

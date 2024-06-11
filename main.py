import os
import tensorflow as tf
import PIL
import tensorflow_hub as hub
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import keras
import keras.utils as image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np
import psycopg2
import time
from sklearn.neighbors import KDTree


my_ip = "http://192.168.95.148:5000" + "/"
conn = psycopg2.connect(database="scanart", user="postgres", password="*", host="127.0.0.1", port='5432')
print("Connection Successful to PostgreSQL")

model = keras.applications.VGG16(weights='imagenet', include_top=True)
styling_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

STYLE_PATHS = {0: ["static/images/Starry Night_770.jpg", "Starry Night"],
               1: ["static/images/Sunflowers_030.jpg", "Sunflowers"],
               2: ["static/images/The Park_958.jpg", "The Park"],
               3: ["static/images/Vitruvian Man_619.jpg", "Vitruvian Man"],
               4: ["static/images/Irises_532.jpg", "Irises"],
               5: ["", "None"]}

# print(model.summary())

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img_for_styling(path_to_img):
    max_dim = 768
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
with conn.cursor() as cursor:
    cursor.execute("SELECT path, features from images")
    res = cursor.fetchall()
images = [x[0] for x in res]
features = [x[1] for x in res]
tree = KDTree(features, metric="euclidean")

for i in range(len(features)):
    if len(features[i]) != 4096:
        print(images[i])


app = Flask(__name__)


@app.route("/recognize_image/<device_id>", methods=['POST'])
def recognize_image(device_id):
    # print(request.json['image'])
    im = Image.open(BytesIO(base64.b64decode(request.json['image'])))
    ts = str(int(time.time()))
    im.save("tmp/img" + ts + ".jpg")
    new_image, x = load_image("tmp/img" + ts + ".jpg")
    new_features = feat_extractor.predict(x)[0]

    # project it into pca space
    # new_pca_features = pca.transform(new_features)[0]
    # print(new_pca_features)

    # bruteforce
    # distances = [distance.cosine(new_features, feat) for feat in features]
    # closest = sorted(range(len(distances)), key=lambda k: distances[k])[0]  # grab first
    # path = images[closest]

    # tree
    dist, ind = tree.query([new_features], 1)
    path = images[ind[0][0]]
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM images where path='{path}'")
        loaded_img = cursor.fetchone()
        cursor.execute(f"SELECT name FROM authors WHERE id={loaded_img[2]}")
        author_name = cursor.fetchone()
        cursor.execute(f"SELECT * FROM labels WHERE image_id={loaded_img[8]}")
        labels = cursor.fetchall()
        labels = [{'x': label[1], 'y': label[2], 'text': label[3], 'id': label[4]} for label in labels]
        cursor.execute(f"SELECT * FROM images WHERE author_id={loaded_img[2]} ORDER BY RANDOM() LIMIT 8")
        carousel = cursor.fetchall()
        carousel = [{"name": rec[0],
                     "uri": my_ip + rec[4],
                     'id': rec[8]
                     } for rec in carousel]
        cursor.execute(f"SELECT * FROM bookmarks where image_id={loaded_img[8]} AND device_id='{device_id}'")
        bookmark = bool(cursor.fetchone())

        cursor.execute(f"update user_achievements set progress=progress+1 where device_id='{device_id}' and "
                       f"achievement_id <= 6")
    conn.commit()

    return jsonify({"name": loaded_img[0],
                    "id": loaded_img[8],
                    "uri": my_ip + loaded_img[4],
                    'author': author_name,
                    "description": loaded_img[1],
                    'labels': labels,  # {x: int, y: int, text: str}
                    'bookmark': bookmark,
                    'carousel': carousel,
                    'width': loaded_img[5],
                    'height': loaded_img[6],
                    'year': loaded_img[3]
                    })


@app.route('/get_image_by_id/<int:image_id>/<device_id>')
def get_image_by_id(image_id, device_id):
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM images where id='{image_id}'")
        loaded_img = cursor.fetchone()
        cursor.execute(f"SELECT name FROM authors WHERE id={loaded_img[2]}")
        author_name = cursor.fetchone()
        cursor.execute(f"SELECT * FROM labels WHERE image_id={loaded_img[8]}")
        labels = cursor.fetchall()
        labels = [{'x': label[1], 'y': label[2], 'text': label[3], 'id': label[4]} for label in labels]
        cursor.execute(f"SELECT * FROM images WHERE author_id={loaded_img[2]} AND id<>{image_id} ORDER BY RANDOM() LIMIT 8")
        carousel = cursor.fetchall()
        carousel = [{"name": rec[0],
                     "uri": my_ip + rec[4],
                     'id': rec[8]
                     } for rec in carousel]
        cursor.execute(f"SELECT * FROM bookmarks where image_id={loaded_img[8]} AND device_id='{device_id}'")
        bookmark = bool(cursor.fetchone())
    # print(loaded_img)
    return jsonify({"name": loaded_img[0],
                    "id": loaded_img[8],
                    "uri": my_ip + loaded_img[4],
                    'author': author_name,
                    "description": loaded_img[1],
                    'labels': labels,  # {x: int, y: int, text: str}
                    'bookmark': bookmark,
                    'carousel': carousel,
                    'width': loaded_img[5],
                    'height': loaded_img[6],
                    'year': loaded_img[3]
                    })


@app.route('/toggle_bookmark/<device_id>/<int:image_id>')
def toggle_bookmark(device_id, image_id):
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM bookmarks where device_id='{device_id}' AND image_id={image_id}")
        bookmark = cursor.fetchone()
        if bookmark:
            cursor.execute(f"DELETE FROM bookmarks where device_id='{device_id}' AND image_id={image_id}")
        else:
            cursor.execute(f"INSERT INTO bookmarks (device_id, image_id) VALUES ('{device_id}', {image_id})")
    conn.commit()
    return {'status': 'ok'}


@app.route('/get_bookmarks/<device_id>')
def get_bookmarks(device_id):
    with conn.cursor() as cursor:
        cursor.execute(
            f"select * from images where id in (SELECT (image_id) FROM bookmarks where device_id='{device_id}') ORDER BY id"
        )
        bookmarks = cursor.fetchall()

    au_ids = [x[2] for x in bookmarks if x[2]]
    authors_dict = {x: '' for x in au_ids}
    with conn.cursor() as cursor:
        if au_ids:
            cursor.execute(
                f"select * from authors where id in {tuple(au_ids) if len(au_ids) > 1 else '(' + str(au_ids[0]) + ')'}")
            pairs = cursor.fetchall()
            for pair in pairs:
                authors_dict[pair[1]] = pair[0]
    bookmarks = [{"name": rec[0],
                  "uri": my_ip + rec[4],
                  'id': rec[8],
                  'author': authors_dict[rec[2]],
                  } for rec in bookmarks]
    return jsonify(bookmarks)


@app.route("/get_profile/<device_id>")
def get_profile(device_id):
    with conn.cursor() as cursor:
        cursor.execute(f"select * from users where device_id='{device_id}'")
        user = cursor.fetchone()
        if not user:
            cursor.execute(
                f"insert into users (device_id, name, label_score) values ('{device_id}', 'artlover', 0)")
            conn.commit()
            user = (device_id, 'artlover', 0)
        cursor.execute("select * from achievements order by id")
        ach = cursor.fetchall()
        cursor.execute(f"select progress from user_achievements where device_id='{device_id}' order by achievement_id")
        user_ach = cursor.fetchall()
        if len(user_ach) < len(ach):
            for achievement in ach[-(len(ach) - len(user_ach)):]:
                cursor.execute(
                    f"insert into user_achievements (device_id, achievement_id, progress) values "
                    f"('{device_id}', {achievement[0]}, 0)")
                print(f"insert into user_achievements (device_id, achievement_id, progress) values "
                      f"('{device_id}', {achievement[0]}, 0)")
            conn.commit()
            user_ach = [[0] for _ in range(len(ach))]
        ach_result = [{'title': ach[k][1],
                       'description': ach[k][2],
                       'required': ach[k][3],
                       'pic_path': ach[k][4],
                       'progress': user_ach[k][0],
                       'id': ach[k][0]} for k in range(len(ach))]
        cursor.execute(f"select * from users order by label_score desc")
        users = cursor.fetchall()
    users = [{"id": user[3], "name": user[1], "score": user[2]} for user in users]
    return jsonify({"achievements": ach_result,
                    "label_score": user[2],
                    "leaderboard": users})


@app.route("/suggest_label", methods=['POST'])
def suggest_label():
    with conn.cursor() as cursor:
        cursor.execute(
            "insert into suggested_labels (device_id, x, y, image_id, text) values (%s, \
    %s, %s, %s, %s)", (request.json['device_id'], request.json['x'], request.json['y'], request.json['image_id'],
                       request.json['text']))
    conn.commit()
    return "ok"


@app.route("/search_images/<query>")
def search_images(query):
    with conn.cursor() as cursor:
        cursor.execute("select * from images where LOWER(name) like LOWER(%s) order by id limit 30", ('%' + query + '%',))
        results = cursor.fetchall()

    au_ids = [x[2] for x in results if x[2]]
    authors_dict = {x: '' for x in au_ids}
    with conn.cursor() as cursor:
        if au_ids:
            cursor.execute(
                f"select * from authors where id in {tuple(au_ids) if len(au_ids) > 1 else '(' + str(au_ids[0]) + ')'}")
            pairs = cursor.fetchall()
            for pair in pairs:
                authors_dict[pair[1]] = pair[0]
    results = [{"name": rec[0],
                "uri": my_ip + rec[4],
                'id': rec[8],
                'author': authors_dict[rec[2]],
                } for rec in results]
    return jsonify(results)


@app.route("/paint", methods=['POST'])
def paint():
    im = Image.open(BytesIO(base64.b64decode(request.json['image'])))
    ts = str(int(time.time()))
    im.save("tmp/to_style" + ts + ".jpg")
    style_id = request.json["style_id"]
    style_path = STYLE_PATHS[style_id][0]
    content_image = load_img_for_styling("tmp/to_style" + ts + ".jpg")
    style_image = load_img_for_styling(style_path)
    stylized_image = styling_model(tf.constant(content_image), tf.constant(style_image))[0]
    pil_img = tensor_to_image(stylized_image)
    pil_img.save("static/generated/styled" + ts + ".jpg")
    width, height = pil_img.size
    return jsonify({"uri": my_ip + "static/generated/styled" + ts + ".jpg",
                    "style_name": STYLE_PATHS[style_id][1],
                    "width": width,
                    "height": height})


@app.route("/change_name", methods=["POST"])
def change_name():
    device_id, name = request.json["device_id"], request.json["name"]
    with conn.cursor() as cursor:
        cursor.execute("UPDATE users SET name=%s where device_id=%s", (name, device_id))
    conn.commit()
    return "200"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

# http://192.168.0.103:5000/

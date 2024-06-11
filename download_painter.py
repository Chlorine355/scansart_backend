from pprint import pprint
import time
import requests
from bs4 import BeautifulSoup
from PIL import ImageFile
import psycopg2

conn = psycopg2.connect(database="scanart", user="postgres", password="*", host="127.0.0.1", port='5432')
cursor = conn.cursor()
print("Connection Successful to PostgreSQL")

start = time.time()
author_url = input("Введите ссылку на автора на  gallerix: ")
author_page = requests.get(author_url, headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 YaBrowser/23.7.5.734 Yowser/2.5 Safari/537.36'})
soup = BeautifulSoup(author_page.content, "html.parser")
author_name = soup.select(".panel-title b")[0].text if soup.select(".panel-title b") else soup.select(".panel-title")[
    0].text
images = []
cursor.execute(f"INSERT INTO authors (name) VALUES ('{author_name}') RETURNING id")
author_id = cursor.fetchone()[0]
print(author_id)
conn.commit()
links = ["https://gallerix.org" + link["href"] for link in soup.select('.pic a.animsition-link')]
for i in range(len(links)):
    img_data = requests.get(links[i], headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 YaBrowser/23.7.5.734 Yowser/2.5 Safari/537.36'})
    soup = BeautifulSoup(img_data.content, "html.parser")
    for x in soup.select("h1.panel-title *"):
        x.decompose()
    image = {"name": soup.select("h1.panel-title")[0].text.strip().replace("<", "").replace(">", "").replace(":", "").replace('"', "")
              .replace("/", "").replace("\\", "").replace("|", "").replace("?", "").replace("*", "").replace("", ""), "source": soup.select("#axpic img")[0]["src"],
             "postfix": soup.select("#axpic img")[0]["src"][-7:-4].replace("/", "")}
    images.append(image)
    print("added {}/{}".format(i, len(links)))

for i in range(len(images)):
    img_data = requests.get(images[i]["source"], headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 YaBrowser/23.7.5.734 Yowser/2.5 Safari/537.36'}).content
    p = ImageFile.Parser()
    p.feed(img_data)  ## feed the data to image parser to get photo info from data headers
    if p.image:
        images[i]["width"], images[i]["height"] = p.image.size
    with open('static/images/' + images[i]["name"] + '_' +
              images[i]["postfix"] + '.jpg', 'wb') as handler:
        handler.write(img_data)
        print("downloaded {}/{}".format(i, len(images)))
    print(images[i])
    cursor.execute(
        f"INSERT INTO images (name, author_id, path, width, height) VALUES ('{images[i]['name']}', {author_id}, '{'static/images/' + images[i]['name'] + '_' + images[i]['postfix'] + '.jpg'}', {images[i]['width']}, {images[i]['height']}) RETURNING id")
conn.commit()
print("На загрузку", len(images), "ушло", time.time() - start)

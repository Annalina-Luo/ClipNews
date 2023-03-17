from pymongo import MongoClient
import os
import json

client = MongoClient(host='localhost', port=27017)
db = client.goodnews


def ann_json(mode, image_path):
    count = 0
    news_list = []

    for split in db.splits.find({'split': mode}):
        print(count)
        if os.path.exists(os.path.join(image_path, f"{split['_id']}.jpg")) == False:
            continue
        news = {}
        article_id = split["article_id"]
        image_id = split["image_index"]
        article = db.articles.find_one({'_id': article_id})
        news["id"] = split["_id"]
        news["image_path"] = split["_id"]+".jpg"
        if mode != 'test':
            news["caption"] = article["images"][image_id].strip()
            news["article"] = article["context"].strip().replace("\n", " ")
        news_list.append(news)
        count += 1

    print(len(news_list))
    with open(f"./{mode}.json", "w") as f:
        json.dump(news_list, f)

    return news_list


def gts_json(mode, image_path):
    count = 0
    news_list = []

    for split in db.splits.find({'split': mode}):
        print(count)
        if os.path.exists(os.path.join(image_path, f"{split['_id']}.jpg")) == False:
            continue
        news = {}
        article_id = split["article_id"]
        image_id = split["image_index"]
        article = db.articles.find_one({'_id': article_id})
        news["id"] = split['_id']
        news["caption"] = article["images"][image_id].strip()
        news_list.append(news)
        count += 1

    print(len(news_list))
    with open(f"./{mode}_gts.json", "w") as f:
        json.dump(news_list, f)

    return news_list


if __name__ == "__main__":
    image_path = "F:/NLP\\transform-and-tell\\data\\goodnews\\goodnews\\images_processed\\"
    # ann_json("train", image_path)
    # ann_json("val", image_path)
    # ann_json("test", image_path)
    gts_json("val", image_path)

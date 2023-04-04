from pymongo import MongoClient
import os
import json
import argparse
from tqdm import tqdm

client = MongoClient(host='localhost', port=27017)
db = client.goodnews  # connect the goodnews database


def ann_json(mode, image_path):
    """
    input:
        mode: train/test/val
        image_path: the path to the image_processed file

    return:
        a json file contains the news list. Each item is a dic of a news.
    """
    count = 0
    news_list = []  # create an empty list to store the news items

    # loop over the splits in the database for the given mode
    for split in tqdm(db.splits.find({'split': mode})):
        # print(count)
        if os.path.exists(os.path.join(image_path, f"{split['_id']}.jpg")) == False:
            # checks if the image associated with each article exists in the image_path directory
            continue
        news = {}
        article_id = split["article_id"]
        image_id = split["image_index"]
        article = db.articles.find_one({'_id': article_id})
        # set the ID of the news item to the split ID
        news["id"] = split["_id"]
        # set the image path to the filename of the image
        news["image_path"] = split["_id"]+".jpg"
        if mode != 'test':
            # if not in test mode, add the caption and article text
            # set the caption to the corresponding image caption in the article
            news["caption"] = article["images"][image_id].strip().replace("\n", " ")
            # set the article text to the context of the article
            news["article"] = article["context"].strip().replace("\n", " ")
        news_list.append(news)
        count += 1

    # write the news list to a JSON file
    with open(f"./{mode}.json", "w") as f:
        json.dump(news_list, f)

    return news_list


def gts_json(mode, image_path):
    """
    Obtain the groudtruth only

    input:
        mode: train/test/val
        image_path: the path to the image_processed file

    return:
        a json file contains the news list. Each item is a dic of a news.
    """
    count = 0
    news_list = []  # create an empty list to store the news items

    for split in tqdm(db.splits.find({'split': mode})):
        # print(count)
        if os.path.exists(os.path.join(image_path, f"{split['_id']}.jpg")) == False:
            # check if the image file exists
            continue
        news = {}
        article_id = split["article_id"]
        image_id = split["image_index"]
        article = db.articles.find_one({'_id': article_id})
        # set the ID of the news item to the split ID
        news["id"] = split['_id']
        # set the caption to the corresponding image caption in the article
        news["caption"] = article["images"][image_id].strip().replace("\n", " ")
        news_list.append(news)
        count += 1

    print("There are totally ", count, "items")
    # write the news list to a JSON file
    with open(f"./{mode}_gts.json", "w") as f:
        json.dump(news_list, f)

    return news_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        default='./images_processed\\')
    args = parser.parse_args()

    # obtain the training data
    ann_json("train", args.image_path)
    print("Finish processing training data.")

    # # obtain the validation data
    ann_json("val", args.image_path)
    print("Finish processing validation data.")

    # # obtain the testing data
    ann_json("test", args.image_path)
    print("Finish processing testing data.")

    # obtain the val groudtruth data
    gts_json("val", args.image_path)
    print("Finish processing validation groudtruth data.")

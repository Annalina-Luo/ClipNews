from PIL import Image
import skimage.io as io
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
import os
import numpy as np
import clip
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
text_model = RobertaModel.from_pretrained('roberta-base')


class NewsDataset(data.Dataset):

    def __init__(self, image_dir, ann_path, preprocess):

        self.image_dir = image_dir
        self.ann = json.load(open(ann_path, 'r'))
        self.preprocess = preprocess

    def __getitem__(self, index):
        # Image
        image_path = os.path.join(
            self.image_dir, self.ann[index]['image_path'])
        image = io.imread(image_path)
        pil_image = Image.fromarray(image)
        image_input = self.preprocess(pil_image)
        # image_input [3, 224, 224]

        # Caption
        caption = self.ann[index]['caption']
        caption = text_tokenizer(
            caption, add_prefix_space=True, return_tensors='pt')
        caption_ids = caption["input_ids"]
        caption_mask = caption["attention_mask"]
        caption_embedding = text_model(**caption)["last_hidden_state"].detach()

        # Article
        article = self.ann[index]['article'][:800]
        article = text_tokenizer(
            article, add_prefix_space=True, return_tensors='pt')

        article_ids = article["input_ids"]
        article_mask = article["attention_mask"]
        article_embedding = text_model(**article)["last_hidden_state"].detach()

        return image_input, caption_ids, caption_mask, caption_embedding, self.ann[index]['id'], article_ids, article_mask, article_embedding

    def __len__(self):
        return len(self.ann)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, article,...).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 224, 224).
            - caption_ids: (len)
            - caption_mask: (len)
            - caption_embedding:
            - caption: torch tensor of shape (?); variable length.
            - article_ids
            - article_mask
            - article_embedding
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        target_ids: torch tensor of shape (batch_size, padded_length)
        target_mask: torch tensor of shape (batch_size, padded_length) 
        target_embedding, 
        lengths: list; valid length for each padded caption.
        # ids: tupl
        target1_ids
        target1_mask
        target1_embedding
        lengths1: list; valid length for each article.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions_ids, captions_mask, captions_embedding, ids, articles_ids, articles_mask, articles_embedding = zip(
        *data)

    # Merge images
    images = torch.stack(list(images))
    # images [batch_size, 3, 224, 224]

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [cap.shape[1] for cap in captions_ids]
    target_ids = torch.zeros(len(captions_ids), max(lengths))
    for i, cap in enumerate(captions_ids):
        end = lengths[i]
        target_ids[i, :end] = cap[0]
    target_mask = torch.zeros(len(captions_mask), max(lengths))
    for i, cap in enumerate(captions_mask):
        end = lengths[i]
        target_mask[i, :end] = cap[0]
    target_embedding = torch.zeros(len(captions_mask), max(lengths), 768)
    for i, cap in enumerate(captions_embedding):
        end = lengths[i]
        target_embedding[i, :end, :] = cap[0][:, :]

    # Article
    lengths1 = [article.shape[1] for article in articles_mask]
    target1_ids = torch.zeros(len(articles_ids), max(lengths1))
    for i, article in enumerate(articles_ids):
        end = lengths1[i]
        target1_ids[i, :end] = article[0]
    target1_mask = torch.zeros(len(articles_mask), max(lengths1))
    for i, article in enumerate(articles_mask):
        end = lengths1[i]
        target1_mask[i, :end] = article[0]
    target1_embedding = torch.zeros(len(articles_mask), max(lengths1), 768)
    for i, article in enumerate(articles_embedding):
        end = lengths1[i]
        target1_embedding[i, :end, :] = article[0][:, :]

    return images, target_ids, target_mask, target_embedding, lengths, ids, target1_ids, target1_mask, target1_embedding, lengths1

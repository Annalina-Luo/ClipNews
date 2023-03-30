# -*- coding: UTF-8 -*-
import pickle
import time
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from model import Encoder_text, Decoder, NewsTransformer, bleu, ciderScore, CLIP_encoder
import os
from dataloader import NewsDataset, collate_fn
# from nltk.translate.bleu_score import corpus_bleu
from build_vocab import Vocabulary
import numpy as np
from utils import *
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import spacy
import random
import math


# Device configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type=str, default='ClipNews_GoodNews')
parser.add_argument('--model_path', type=str,
                    default='.\\model_save\\', help='path for saving trained models')
parser.add_argument('--image_size', type=int, default=256,
                    help='size for resize images')
parser.add_argument('--crop_size', type=int, default=224,
                    help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str,
                    default='.\\vocab.pkl', help='path for vocabulary wrapper')
# parser.add_argument('--vocab1_path', type=str,
#                     default='.\\articale_vocab.pkl', help='article vocab')
parser.add_argument('--image_dir', type=str,
                    default='F:\\NLP\\transform-and-tell\\data\\goodnews\\goodnews/images_processed/', help='directory for resized images')
parser.add_argument('--ann_path', type=str, default='./',
                    help='path for annotation json file')
parser.add_argument('--log_step', type=int, default=100,
                    help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000,
                    help='step size for saving trained models')
# {image_id': image_id, 'caption': " ".join(preds)}
parser.add_argument('--gts_file_dev', type=str, default='./val_gts.json')

# Model parameters
parser.add_argument('--embed_dim', type=int, default=768,
                    help='dimension of word embedding vectors')
# parser.add_argument('--attention_dim', type=int, default=512,
#                     help='dimension of attention linear layers')
# parser.add_argument('--decoder_dim', type=int, default=512,
#                     help='dimension of decoder rnn')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--epochs_since_improvement', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--encoder_lr', type=float, default=0.0005)
parser.add_argument('--decoder_lr', type=float, default=0.0005)
parser.add_argument('--checkpoint', type=str, default=None,
                    help='path for checkpoints')
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--best_cider', type=float, default=0.)
# parser.add_argument('--fine_tune_encoder', type=bool,
# default=False, help='fine-tune encoder')
# parser.add_argument('--image_encoder', type=bool,
#                     default="ResNet", help='Image encoder [ResNet | CLIP]')
# parser.add_argument('--word_embedding', type=bool,
# default="GPT2", help='Image encoder [RoBERTa | word2vector]')

args = parser.parse_args()


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main(args):

    global best_cider, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, train_logger, dev_logger

    # fine_tune_encoder = args.fine_tune_encoder

    if args.checkpoint is None:
        enc_text = Encoder_text(args.embed_dim, 1, 8, 512, 0.1)
        dec = Decoder(args.embed_dim, 2, 8, 512, 0.1)
        ImageEncoder = CLIP_encoder(args.embed_dim)
        # print(type(ImageEncoder))
        # print(type(enc_text))
        model = NewsTransformer(enc_text, ImageEncoder,
                                dec, args.embed_dim, 0, 0)
        optimizer = optim.Adam(model.parameters(), lr=args.decoder_lr)

        # ImageEncoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, ImageEncoder.parameters()),
                                             lr=args.encoder_lr)

        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        model.apply(initialize_weights)
        start_epoch = args.start_epoch

    else:
        checkpoint = torch.load(args.checkpoint)
        # print(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        # print(start_epoch)
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        # optimizer = checkpoint['optimizer']
        model = checkpoint['decoder']
        optimizer = optim.Adam(model.parameters(), lr=args.decoder_lr)

        encoder_optimizer = checkpoint['encoder_optimizer']
        # ImageEncoder = checkpoint['ImageEncoder']
        # if fine_tune_encoder is True:
        # ImageEncoder.module.fine_tune(fine_tune_encoder)
        if encoder_optimizer is None:
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, ImageEncoder.parameters()),
                                                 lr=args.encoder_lr)

    model = model.to(device)

    # ImageEncoder = ImageEncoder.to(device)
    # if torch.cuda.device_count() > 1 and not isinstance(ImageEncoder, torch.nn.DataParallel):
    #     ImageEncoder = torch.nn.DataParallel(ImageEncoder)

    train_log_dir = os.path.join(args.model_path, 'train')
    dev_log_dir = os.path.join(args.model_path, 'dev')
    train_logger = Logger(train_log_dir)
    dev_logger = Logger(dev_log_dir)

    criterion = nn.CrossEntropyLoss().to(device)

    train_ann_path = os.path.join(args.ann_path, 'train_s.json')
    train_data = NewsDataset(args.image_dir, train_ann_path)
    print('train set size: {}'.format(len(train_data)))
    # batch
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=collate_fn)

    dev_ann_path = os.path.join(args.ann_path, 'val.json')
    dev_data = NewsDataset(args.image_dir, dev_ann_path)
    print('dev set size: {}'.format(len(dev_data)))
    val_loader = torch.utils.data.DataLoader(dataset=dev_data, batch_size=1, shuffle=False,
                                             num_workers=args.num_workers, collate_fn=collate_fn)

    best_cider = args.best_cider
    for epoch in range(start_epoch, args.epochs):

        print(epoch)
        if args.epochs_since_improvement == 20:
            break
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 6 == 0:
            adjust_learning_rate(optimizer, 0.6)

        # print("start training...")
        train(model=model,
              train_loader=train_loader,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              optimizer=optimizer,
              epoch=epoch,
              logger=train_logger,
              logging=True)
        # print("finish training...")

        if epoch > 4:
            recent_cider = validate(model=model,
                                    val_loader=val_loader,
                                    criterion=criterion,
                                    epoch=epoch,
                                    logger=dev_logger,
                                    logging=True)

            is_best = recent_cider > best_cider
            best_cider = max(recent_cider, best_cider)
            print('best_cider:', best_cider)
            print('learning_rate:', args.decoder_lr)
            if not is_best:
                args.epochs_since_improvement += 1
                print("\nEpoch since last improvement: %d\n" %
                      (args.epochs_since_improvement,))
            else:
                args.epochs_since_improvement = 0

        if epoch <= 4:
            recent_cider = 0
            is_best = 1

        save_checkpoint(args.data_name, epoch, args.epochs_since_improvement,
                        model, encoder_optimizer, optimizer, recent_cider, is_best)


def train(model, train_loader, encoder_optimizer, optimizer, criterion, epoch, logger, logging=True):
    # ImageEncoder.train()
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top5accs = AverageMeter()

    start = time.time()

    t = tqdm(train_loader, desc='Train %d' % epoch)

    for i, (imgs, caps_ids, caps_mask, caps_emb, caplens, img_ids, arts_ids, arts_mask, arts_emb, artslens) in enumerate(t):
        # print(i)
        # print("imgs", imgs.shape)
        # print("caps_mask", caps_mask.shape)
        # print("arts_mask", arts_mask.shape)
        # print("caps_emb", caps_emb.shape)
        # print("arts_emb", arts_emb.shape)
        # imgs [batch_size, 3,224, 224]
        # caps [batch_size, len, 768]
        # arts [batch_size, len,768]
        # print("caps", caps.shape)
        # print("arts", arts.shape)
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        caps_ids = caps_ids.to(device)
        caps_mask = caps_mask.to(device)
        caps_emb = caps_emb.to(device)
        arts_ids = arts_ids.to(device)
        arts_mask = arts_mask.to(device)
        arts_emb = arts_emb.to(device)

        # print("start forwad model")
        # output, _, mask = model(arts, caps[:, :-1], imgs)
        output = model(arts_ids, arts_mask, arts_emb,
                       caps_mask, caps_emb, imgs)
        # print("finish forwad model")

        output_dim = output.shape[-1]
        # print("output", output.shape)
        # torch.Size([2944, 50265])
        output = output.contiguous().view(-1, output_dim)
        caps_ids = caps_ids.contiguous().view(-1).long()  # torch.Size([2944])
        # print(output.shape, caps_ids.shape)

        loss = criterion(output, caps_ids)

        optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        decode_lengths = [c - 2 for c in caplens]
        losses.update(loss.item(), sum(decode_lengths))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        batch_time.update(time.time() - start)

        start = time.time()

    # log into tf series
    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch,
          args.epochs, losses.avg, np.exp(losses.avg)))
    if logging:
        logger.scalar_summary('loss', losses.avg, epoch)
        logger.scalar_summary('Perplexity', np.exp(losses.avg), epoch)


def validate(model, val_loader, criterion, epoch, logger, logging=True):
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    res = []
    start = time.time()

    # Batches
    t = tqdm(val_loader, desc='Dev %d' % epoch)
    for i, (imgs, caps_ids, caps_mask, caps_emb, caplens, img_ids, arts_ids, arts_mask, arts_emb, artslens) in enumerate(t):
        # if i ==10:
        # break
        # Move to device, if available
        print("imgs", imgs.shape)
        print("caps_mask", caps_mask.shape)
        print("arts_mask", arts_mask.shape)
        print("caps_emb", caps_emb.shape)
        print("arts_emb", arts_emb.shape)
        imgs = imgs.to(device)
        caps_ids = caps_ids.to(device)
        caps_mask = caps_mask.to(device)
        caps_emb = caps_emb.to(device)
        arts_ids = arts_ids.to(device)
        arts_mask = arts_mask.to(device)
        arts_emb = arts_emb.to(device)
        # imgs = ImageEncoder(imgs)

        print("\n start validation")
        output = model(arts_ids, arts_mask, arts_emb,
                       caps_mask, caps_emb, imgs)
        print("finish validation")
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        caps_ids = caps_ids.contiguous().view(-1).long()
        # print(output.shape)  # torch.Size([3264, 50265])
        # print(caps_ids.shape)  # torch.Size([3264])
        loss = criterion(output, caps_ids)

        decode_lengths = [c - 2 for c in caplens]
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        # outputs = model.sample(caps, caplens, arts, mask, model)
        print("start bleu")
        outputs = bleu(model, arts_ids, arts_mask,
                       arts_emb, caplens, imgs, device)
        print(outputs)

        preds = outputs

        for idx, image_id in enumerate(img_ids):
            res.append({'image_id': image_id, 'caption': " ".join(preds)})

    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch,
          args.epochs, losses.avg, np.exp(losses.avg)))
    # gts_file = args.gts_file_dev
    # eval_out = myeval(res, gts_file)

    score = ciderScore(args.gts_file_dev, res)

    if logging:
        logger.scalar_summary(score, "Cider", epoch)
    # log into tf series
    if logging:
        logger.scalar_summary('loss', losses.avg, epoch)
        logger.scalar_summary('Perplexity', np.exp(losses.avg), epoch)
    return score


if __name__ == '__main__':
    main(args)

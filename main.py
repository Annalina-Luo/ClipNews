import time
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from model import Encoder_text, Decoder, NewsTransformer, translate_sentence, ciderScore, CLIP_encoder
import os
from dataloader import NewsDataset, collate_fn
import numpy as np
from utils import *
import torch.optim as optim
import clip


# Device configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
clip_model.to(device)
parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type=str,
                    default='ClipNews_GoodNews', help="the name of the dataset.")
parser.add_argument('--model_path', type=str,
                    default='.\\model_save\\', help='path for saving trained models')
parser.add_argument('--image_dir', type=str,
                    default='./images_processed/', help='directory for resized images')
parser.add_argument('--ann_path', type=str, default='/mnt/',
                    help='path for annotation json file')
parser.add_argument('--log_step', type=int, default=100,
                    help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000,
                    help='step size for saving trained models')
parser.add_argument('--gts_file_dev', type=str, default='/mnt/train_gts.json')

# Model parameters
parser.add_argument('--embed_dim', type=int, default=768,
                    help='dimension of word embedding vectors')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--start_epoch', type=int, default=0,
                    help="the starting epoch.")
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--epochs_since_improvement', type=int, default=0,
                    help="the number of epochs since the last improvement in validation loss.")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--encoder_lr', type=float, default=0.0005)
parser.add_argument('--decoder_lr', type=float, default=0.0005)
parser.add_argument('--checkpoint', type=str, default=None,
                    help='path for checkpoints')
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--best_cider', type=float, default=0.,
                    help=" the best CIDEr score achieved during training.")
parser.add_argument('--ImageEncoder_attention', type=bool, default=False,
                    help="whether add attention module in the image encoder")
parser.add_argument('--TextEncoder_attention', type=bool, default=False,
                    help="whether add attention module in the article encoder")


args = parser.parse_args()


def get_parameter_number(net):
    # returns the total number of parameters and trainable parameters in the model.
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main(args):
    # Initializing global variables
    global best_cider, epochs_since_improvement, checkpoint, start_epoch, data_name

    if args.checkpoint is None:
        enc_text = Encoder_text(args.embed_dim, 1, 8,
                                512, 0.1, args.TextEncoder_attention)
        dec = Decoder(args.embed_dim, 2, 8, 512, 0.1)
        ImageEncoder = CLIP_encoder(
            args.embed_dim, 1, 8, 512, 0.1, clip_model, args.ImageEncoder_attention)
        model = NewsTransformer(enc_text, ImageEncoder,
                                dec, args.embed_dim, 0, 0)

        optimizer = optim.Adam(model.parameters(), lr=args.decoder_lr)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, ImageEncoder.parameters()),
                                             lr=args.encoder_lr)

        # Initializing model weights
        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        model.apply(initialize_weights)
        start_epoch = args.start_epoch
        best_cider = args.best_cider

    else:
        # Loading model from checkpoint
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        ImageEncoder = checkpoint['ImageEncoder']
        enc_text = checkpoint['enc_text']
        dec = checkpoint['dec']
        model = checkpoint['model']
        best_cider = checkpoint['cider']
        recent_cider = best_cider
        optimizer = optim.Adam(model.parameters(), lr=args.decoder_lr)
        encoder_optimizer = checkpoint['encoder_optimizer']
        if encoder_optimizer is None:
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, ImageEncoder.parameters()),
                                                 lr=args.encoder_lr)
    # Moving model to GPU
    ImageEncoder = ImageEncoder.to(device)
    enc_text = enc_text.to(device)
    dec = dec.to(device)
    model = model.to(device)

    # Defining loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Creating dataloaders
    train_ann_path = os.path.join(args.ann_path, 'test_2.json')
    train_data = NewsDataset(args.image_dir, train_ann_path, preprocess)
    # print('train set size: {}'.format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=collate_fn)

    dev_ann_path = os.path.join(args.ann_path, 'test_2.json')
    dev_data = NewsDataset(args.image_dir, dev_ann_path, preprocess)
    # print('dev set size: {}'.format(len(dev_data)))
    val_loader = torch.utils.data.DataLoader(dataset=dev_data, batch_size=1, shuffle=False,
                                             num_workers=args.num_workers, collate_fn=collate_fn)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        if args.epochs_since_improvement == 20:
            break
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 6 == 0:
            adjust_learning_rate(optimizer, 0.6)

        # Training model
        train(model=model,
              train_loader=train_loader,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              optimizer=optimizer,
              epoch=epoch)

        # Validating model
        if epoch > 5:
            recent_cider = validate(model=model,
                                    val_loader=val_loader,
                                    criterion=criterion,
                                    epoch=epoch)

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
                        ImageEncoder, enc_text, dec, model, encoder_optimizer, optimizer, recent_cider, is_best)


def train(model, train_loader, encoder_optimizer, optimizer, criterion, epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    t = tqdm(train_loader, desc='Train %d' % epoch)

    for i, (imgs, caps_ids, caps_mask, caps_emb, caplens, img_ids, arts_ids, arts_mask, arts_emb, artslens) in enumerate(t):
        # imgs [batch_size, 3,224, 224]
        # caps_ids [batch_size, cap_len]
        # caps_emb [batch_size, cap_len, 768]
        # arts_mask [batch_size, art_len]
        # arts_emb [batch_size, art_len, 768]

        # measure the time it takes to load the data
        data_time.update(time.time() - start)

        # move the tensors to the specified device (CPU or GPU)
        imgs = imgs.to(device)
        caps_ids = caps_ids.to(device)
        caps_mask = caps_mask.to(device)
        caps_emb = caps_emb.to(device)
        arts_ids = arts_ids.to(device)
        arts_mask = arts_mask.to(device)
        arts_emb = arts_emb.to(device)

        output = model(arts_ids, arts_mask, arts_emb,
                       caps_mask[:, :-1], caps_emb[:, :-1, :], imgs)  # compute the output of the model

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        caps_ids = caps_ids[:, 1:].contiguous(
        ).view(-1).long()  # torch.Size([2944])

        loss = criterion(output, caps_ids)  # compute the loss

        optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        decode_lengths = [c - 1 for c in caplens]
        # update the losses meter
        losses.update(loss.item(), sum(decode_lengths))

        loss.backward()  # backpropagate the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()  # update the model parameters
        batch_time.update(time.time() - start)

        start = time.time()

    # log into tf series
    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch,
          args.epochs, losses.avg, np.exp(losses.avg)))
    # if logging:
    #     logger.scalar_summary('loss', losses.avg, epoch)
    #     logger.scalar_summary('Perplexity', np.exp(losses.avg), epoch)


def validate(model, val_loader, criterion, epoch):
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    res = []  # Initialize an empty list to store the results
    start = time.time()

    # Batches
    t = tqdm(val_loader, desc='Dev %d' % epoch)
    for i, (imgs, caps_ids, caps_mask, caps_emb, caplens, img_ids, arts_ids, arts_mask, arts_emb, artslens) in enumerate(t):
        # Move the inputs to the device
        imgs = imgs.to(device)
        caps_ids = caps_ids.to(device)
        caps_mask = caps_mask.to(device)
        caps_emb = caps_emb.to(device)
        arts_ids = arts_ids.to(device)
        arts_mask = arts_mask.to(device)
        arts_emb = arts_emb.to(device)

        # Compute the model's output and loss
        output = model(arts_ids, arts_mask, arts_emb,
                       caps_mask[:, :-1], caps_emb[:, :-1, :], imgs)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        caps_ids = caps_ids[:, 1:].contiguous().view(-1).long()
        loss = criterion(output, caps_ids)

        # Compute the decode lengths and update the loss and batch time
        decode_lengths = [c - 1 for c in caplens]
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        # Compute the predictions for the inputs
        # outputs = bleu(model, arts_ids, arts_mask,
        #                arts_emb, caplens, imgs, device)
        prediction = translate_sentence(
            model, arts_ids, arts_mask,
            arts_emb, caplens, imgs, device)

        preds = prediction
        # print("preds", preds)

        # Append the results to the res list
        for idx, image_id in enumerate(img_ids):
            res.append({'image_id': image_id, 'caption': " ".join(preds)})

    # Save the results as a json file
    with open(f"./val_{epoch}.json", "w") as f:
        json.dump(res, f)

    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch,
          args.epochs, losses.avg, np.exp(losses.avg)))

    # Compute the CIDEr score and log it into the logger
    score = ciderScore(args.gts_file_dev, res)

    # if logging:
    #     logger.scalar_summary(score, "Cider", epoch)
    # # # Log the loss and perplexity into the logger
    # if logging:
    #     logger.scalar_summary('loss', losses.avg, epoch)
    #     logger.scalar_summary('Perplexity', np.exp(losses.avg), epoch)
    return score


if __name__ == '__main__':
    main(args)

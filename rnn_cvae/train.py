import os
import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import ImgCapCVAE
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
data_folder = '../data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
save_folder = 'ckpt2/'  # save folder path, for saving different experiment results

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
hidden_size = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 20  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 80
workers = 1  # for data-loading; right now, only 1 works with h5py
finetune_lr = 1e-4  # learning rate for encoder if fine-tuning
cvae_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

# CVAE parameters
latent_size = 128  # gaussian distribution size
standard_gaussian = False  # use standard gaussian (one distribution) or not (two distributions)
k = 1.5  # logistic anneal function slope
xm = 2.  # logistic anneal function mid_value point (when epoch = xm, anneal function = 0.5)
kl_weight = 1.  # kl loss weight (compared to reconstruction loss)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        model = ImgCapCVAE(standard_gaussian=standard_gaussian,
                           attn_dim=attention_dim,
                           embed_dim=emb_dim,
                           hidden_size=hidden_size,
                           latent_size=latent_size,
                           vocab_size=len(word_map))
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        model = checkpoint['model']
    model.imgEncoder.fine_tune(fine_tune_encoder)
    lr = finetune_lr if fine_tune_encoder else cvae_lr
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              optimizer=optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                model=model,
                                epoch=epoch)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(save_folder, data_name, epoch, epochs_since_improvement, model, recent_bleu4, is_best)


def kl_anneal_function(step):
    return float(kl_weight / (1 + np.exp(-k * (step - xm))))


def loss_fn(preds, target, alphas, mu1, logv1, mu2, logv2, step):
    # Reconstruction loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    rec_loss = criterion(preds, target)

    # Attention loss
    attn_loss = ((1. - alphas.sum(dim=1)) ** 2).mean()

    # KL loss
    if standard_gaussian:
        kl_loss = -0.5 * torch.sum(1 + logv1 - mu1.pow(2) - logv1.exp())
    else:
        kl_loss = -0.5 * torch.sum(1 + (logv1 - logv2)
                                   - torch.div(torch.pow(mu2 - mu1, 2), torch.exp(logv2))
                                   - torch.div(torch.exp(logv1), torch.exp(logv2)))
    kl_weight = kl_anneal_function(step)

    return rec_loss, attn_loss, kl_loss, kl_weight


def train(train_loader, model, optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: CVAE model
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    rec_losses = AverageMeter()  # reconstruction losses
    attn_losses = AverageMeter()  # attention losses
    kl_losses = AverageMeter()  # KL losses
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        scores, alphas, mu1, logv1, mu2, logv2, caps_sorted, decode_lengths, sort_ind = model(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        step = float(epoch) + float(i) / len(train_loader)
        rec_loss, attn_loss, kl_loss, kl_w = loss_fn(scores, targets, alphas, mu1, logv1, mu2, logv2, step)
        loss = rec_loss + alpha_c * attn_loss + kl_w * kl_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        rec_losses.update(rec_loss.item(), sum(decode_lengths))
        attn_losses.update(attn_loss.item(), sum(decode_lengths))
        kl_losses.update(kl_loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'rec_loss {rec_loss.avg:.4f}\t'
                  'attn_loss {attn_loss.avg:.4f}\t'
                  'kl_loss {kl_loss.avg:.4f}\t'
                  'kl_weight {kl_weight:.4f}\t'
                  'Top-5 Accuracy {top5.avg:.3f}'.format(epoch, i, len(train_loader),
                                                         batch_time=batch_time,
                                                         rec_loss=rec_losses,
                                                         attn_loss=attn_losses,
                                                         kl_loss=kl_losses,
                                                         kl_weight=kl_w,
                                                         top5=top5accs))
            batch_time.reset()
            rec_losses.reset()
            attn_losses.reset()
            kl_losses.reset()
            top5accs.reset()


def validate(val_loader, model, epoch):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param model: CVAE model
    :param epoch: epoch idx
    :return: BLEU-4 score
    """
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            scores, alphas, mu1, logv1, mu2, logv2, caps_sorted, decode_lengths, sort_ind = model(imgs, caps, caplens,
                                                                                                  val=True)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            # loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            step = float(epoch + 1)
            rec_loss, attn_loss, kl_loss, kl_w = loss_fn(scores, targets, alphas, mu1, logv1, mu2, logv2, step)
            loss = rec_loss + alpha_c * attn_loss + kl_w * kl_loss

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Top-5 Accuracy {top5.avg:.3f}\t'.format(i, len(val_loader), batch_time=batch_time, loss=losses,
                                                               top5=top5accs))
                losses.reset()
                top5accs.reset()
                batch_time.reset()

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()

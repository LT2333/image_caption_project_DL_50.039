import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_word_list_from_idx_list(vocab,traget_idx_list):
    word_list = list()
    for idx in traget_idx_list:
        word_list.append(vocab.idx2word[int(idx[0])])
    return word_list

def main(args):
    print("********Start Evaluation********")   
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.Resize(224), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    encoder = EncoderCNN(args.embed_size).eval().to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).eval().to(device)
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Loss
    criterion = nn.CrossEntropyLoss()

    #define BLEU 1-4 weights for cumulative N-gram scores
    bleu_1_weight = (1,0,0,0)
    bleu_2_weight = (0.5,0.5,0,0)
    bleu_3_weight = (0.33,0.33,0.33,0)
    bleu_4_weight = (0.25,0.25,0.25,0.25)
    cb_1 = list()
    cb_2 = list()
    cb_3 = list()
    cb_4 = list()
    
    #Evaluate the models
    val_losses = list()
    val_accs = 0.0
    total_step = len(data_loader)
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            
            # record accuracy and loss
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())

            topv,topi = outputs.topk(1, dim=1)
            targets = targets.unsqueeze(-1)
            val_accs += float((topi == targets).sum())/targets.shape[0]


            # calculate bleu score
            sentence_length = 0
            for l in range(len(lengths)):
        
                candidate = generate_word_list_from_idx_list(vocab,topi[sentence_length:sentence_length+lengths[l]])
                reference = [generate_word_list_from_idx_list(vocab,targets[sentence_length:sentence_length+lengths[l]])]
                
                cb_1.append(float(sentence_bleu(reference,candidate,weights=bleu_1_weight)))
                cb_2.append(float(sentence_bleu(reference,candidate,weights=bleu_2_weight)))
                cb_3.append(float(sentence_bleu(reference,candidate,weights=bleu_3_weight)))
                cb_4.append(float(sentence_bleu(reference,candidate,weights=bleu_4_weight)))
                
                sentence_length += lengths[l]
                # if i%args.log_step ==0 and l ==61:
                #     print('candidate:',candidate)
                #     print('referrence',reference)
                

            # Print log info
            if i % args.log_step == 0:
                print('Step [{}/{}], Loss: {:.4f},BLEU score:{:4f}/{:4f}/{:4f}/{:4f},Accuracy:{:4f}, Perplexity: {:5.4f}'
                    .format( i+1, total_step, loss.item(), sum(cb_1)/len(cb_1),sum(cb_2)/len(cb_2),sum(cb_3)/len(cb_3),sum(cb_4)/len(cb_4),val_accs/(i+1),np.exp(loss.item()))) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/my-encoder-5-3000-t4-resnext.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/my-decoder-5-3000-t4-resnext.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab_stemmed_t4.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resizedVal2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='../datasets/coco2014/trainval_coco2014_captions/captions_val2014.json', 
                        help='path for val annotation file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    print(args)
    main(args)
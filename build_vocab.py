import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
from nltk.stem.snowball import SnowballStemmer

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.stemmer = SnowballStemmer("english")

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        stemmed_word = self.stemmer.stem(word)
        if not stemmed_word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[stemmed_word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    sb_stemmer = SnowballStemmer("english")

    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokens = [sb_stemmer.stem(word) for word in tokens]
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    print('len: ',len(words))
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    print("count threshold:",args.threshold)
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--caption_path', type=str, 
    #                     default='data/annotations/captions_train2014.json', 
    #                     help='path for train annotation file')
# if on JupyterLab:
    parser.add_argument('--caption_path', type=str, 
                        default='../datasets/coco2014/trainval_coco2014_captions/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_stemmed_t10.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=10, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
import pickle as pkl
import numpy as np
import spacy
import re
import argparse
import pdb

import epitran

from mt_tools import EPITRAN_LANGS


def seed_everything(seed=sum(bytes(b'dragn'))):
    """
    Helper function to set random seed
    """
    #random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True


def g2p(sent, epi):
    """
    Transliterate sentence to IPA using Epitran
    """
    return epi.transliterate(u''+sent)


def clean(src_lines, tgt_lines, lang):    
    # Create epitran transliterator
    epi = epitran.Epitran(EPITRAN_LANGS[lang])
    src_tok, tgt_tok = [], []
    for src_line, tgt_line in zip(src_lines, tgt_lines):
        if re.search('[a-zA-Z]', src_line) and lang in ["th", "lo"]:
            continue
        else:
            sent = src_line
            if sent.strip() != "" and tgt_line.strip() != "":
                try:
                    src_tok.append(g2p(sent, epi))
                    tgt_tok.append(tgt_line)
                except:
                    print("WARNING: Epitran error with sentence", sent)
                    continue
    
    return src_tok, tgt_tok


def write(lines, outfile):
    with open(outfile, "w") as f:
        f.writelines(lines)


def main(src_file, tgt_file, out_file, lang, tgt_lang, train_len, val_len, test_len, \
        num_duplicates=1, seed=0):
    with open(src_file, "r") as f:
        src = f.readlines()
    with open(tgt_file, "r") as f:
        tgt = f.readlines()

    if seed:
        seed_everything(seed)

    # shuffle
    indices = np.arange(len(src))
    if len(src) < train_len + val_len + test_len:
        raise ValueError(f"Train len {train_len} + val len {val_len} + test len {test_len} "\
                          "> the actual length of the bitext {len(src)}")
    np.random.shuffle(indices)
    train, val, test = indices[:train_len], indices[train_len:train_len+val_len], indices[train_len+val_len:train_len+val_len+test_len]

    # split into train/val/test
    src_train, tgt_train = [], []
    for idx in train:
        src_train.append(src[idx])
        tgt_train.append(tgt[idx])

    src_val, tgt_val = [], []
    for idx in val:
        src_val.append(src[idx])
        tgt_val.append(tgt[idx])

    src_test, tgt_test = [], []
    for idx in test:
        src_test.append(src[idx])
        tgt_test.append(tgt[idx])

    print(len(src_train), len(tgt_train))
    # clean and tokenize
    src_train_tok, tgt_train_tok = clean(src_train, tgt_train, lang)
    src_val_tok, tgt_val_tok = clean(src_val, tgt_val, lang)
    src_test_tok, tgt_test_tok = clean(src_test, tgt_test, lang)

    print(len(src_train_tok), len(tgt_train_tok))

    src_train_tok = src_train_tok * num_duplicates
    tgt_train_tok = tgt_train_tok * num_duplicates
    print(len(src_train_tok), len(tgt_train_tok))

    # write to outfile
    write(src_train_tok, out_file + lang + ".train")
    write(tgt_train_tok, out_file + tgt_lang + ".train")

    write(src_val_tok, out_file + lang + ".val")
    write(tgt_val_tok, out_file + tgt_lang + ".val")

    write(src_test_tok, out_file + lang + ".test")
    write(tgt_test_tok, out_file + tgt_lang + ".test")


"""
lo/en split
python3 split.py --src-file en-lo.tgt --tgt-file en-lo.src --out-file ../../OpenNMT-py/th-lo/corpora/en-lo. --lang lo --train-len 15000 --val-len 5000 --test-len 5000
th/en split
python3 split.py --src-file en-th.tgt --tgt-file en-th.src --out-file ../../OpenNMT-py/th-lo/corpora/en-th. --lang th --train-len 250000 --val-len 5000 --test-len 5000
tr/en split
python3 split.py --src-file tr-az/en-tr.tr --tgt-file tr-az/en-tr.en --out-file ../OpenNMT-py/tr-az/corpora/en-tr. --lang tr --train-len 250000 --val-len 5000 --test-len 5000
az/en split
python3 split.py --src-file tr-az/en-az.az --tgt-file tr-az/en-az.en --out-file ../OpenNMT-py/tr-az/corpora/en-az. --lang az --train-len 15000 --val-len 5000 --test-len 5000
ht/en split
python3 split.py --src-file enht_haitian  --tgt-file enht_english --out-file ../OpenNMT-py/fr-ht/corpora/en-ht-15k. --lang ht --train-len 15000 --val-len 5000 --test-len 5000
fr/en split
python3 split.py --src-file new_bitexts/enfr_french --tgt-file new_bitexts/enfr_english --out-file ../OpenNMT-py/fr-ht/corpora/en-fr-250k. --lang fr --train-len 250000 --val-len 5000 --test-len 5000
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file",
                        type=str)
    parser.add_argument("--tgt-file",
                        type=str)
    parser.add_argument("--out-file",
                        type=str)
    parser.add_argument("--lang",
                        type=str,
                        choices=["th", "lo", "tr", "az", "fr", "ht"])
    parser.add_argument("--tgt-lang",
                        type=str,
                        choices=['en', 'fr'],
                        default='en')
    parser.add_argument("--train-len",
                        type=int)
    parser.add_argument("--val-len",
                        type=int)
    parser.add_argument("--test-len",
                        type=int)
    parser.add_argument("--num-duplicates",
                        type=int,
                        default=1)
    args = parser.parse_args()

    # main(src_file, tgt_file, out_file, lang, tgt_lang, train_len, val_len, test_len, \
    #    num_duplicates=1, seed=0)
    main(src_file=args.src_file, tgt_file=args.tgt_file, out_file=args.out_file, lang=args.lang,\
            tgt_lang=args.tgt_lang, train_len=args.train_len, val_len=args.val_len,\
            test_len=args.test_len, num_duplicates=args.num_duplicates)

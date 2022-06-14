import pickle as pkl
import numpy as np
import spacy
#import en_core_web_sm
import re
import argparse
#from pythainlp.tokenize import word_tokenize as thai_word_tokenize
#from laonlp.tokenize import word_tokenize as lao_word_tokenize
#from trtokenizer.tr_tokenizer import WordTokenizer as TurkishWordTokenizer
import pdb

#th_tokenizer = thai_word_tokenize
#lo_tokenizer = lao_word_tokenize
#tr_tokenizer = TurkishWordTokenizer()
#az_tokenizer = spacy.load('xx_ent_wiki_sm')
#ht_tokenizer = spacy.load('xx_ent_wiki_sm')

#def tokenize(sent, lang):
#    if lang == "tha":
#        return " ".join(th_tokenizer(sent)).strip('\n') + " \n"
#    elif lang == "lao":
#        return " ".join(lo_tokenizer(sent)).strip('\n') + " \n"
#    elif lang == "tur": # FIXME
#        return " ".join(tr_tokenizer.tokenize(sent)).strip('\n') + " \n"
#    elif lang == "aze":
#        return " ".join([tok.text for tok in az_tokenizer.tokenizer(sent)]).strip('\n') + " \n"
#    elif lang == "hat":
#        return " ".join([tok.text for tok in ht_tokenizer.tokenizer(sent)]).strip('\n') + " \n" 
#    elif lang == "fra":
#        return " ".join([tok.text for tok in ht_tokenizer.tokenizer(sent)]).strip('\n') + " \n"

def clean(src_lines, tgt_lines, lang):    
    src_tok, tgt_tok = [], []
    for src_line, tgt_line in zip(src_lines, tgt_lines):
        if re.search('[a-zA-Z]', src_line) and lang in ["th", "lo"]:
            continue
        else:
            sent = src_line
            #sent = " ".join(tokenizer(src_line))
            #sent = tokenize(src_line, lang)
            #try:
            #    sent = re.sub("\u200b", "", sent)
            #except:
            #    pdb.set_trace()
            #sent = re.sub(" +", " ", sent)
            if sent.strip() != "" and tgt_line.strip() != "":
                src_tok.append(sent)
                tgt_tok.append(tgt_line)
    
    return src_tok, tgt_tok


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


def write(lines, outfile):
    with open(outfile, "w") as f:
        f.writelines(lines)


def main(src_file, tgt_file, out_file, lang, tgt_lang, train_len, val_len, test_len, seed=0):
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
    parser.add_argument("--train-len",
                        type=int)
    parser.add_argument("--val-len",
                        type=int)
    parser.add_argument("--test-len",
                        type=int)
    args = parser.parse_args()

    main(args.src_file, args.tgt_file, args.out_file, args.lang,\
            args.train_len, args.val_len, args.test_len)   

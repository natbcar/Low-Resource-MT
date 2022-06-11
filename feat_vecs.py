import numpy as np
import epitran
import panphon2

import pdb
import re

ft = panphon2.FeatureTable()
rand_fcn = np.random.random # FIXME

PHON_EMB_LEN = len(ft.word_to_bag_of_features('b'))


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


def pad_vec(vec, emb_dim, phon_info):
    phon_type, phon_pad, ngram_size = phon_info['type'], phon_info['pad'], phon_info['gram']
    if phon_type == 'phon':
        if phon_pad == 'rand':
            return vec + [0] * ((PHON_EMB_LEN * ngram_size) - len(vec)) # + list(rand_fcn(emb_dim - len(vec))) 
        elif phon_pad == 'cat':
            num_repeats = emb_dim // len(vec)
            remainder = emb_dim % len(vec)
            return (vec * num_repeats) + vec[:remainder]
        elif phon_pad == 'zero':
            return vec + ([0] * (emb_dim - len(vec)))
        else:
            raise NotImplementedError("Phon padding str not in {'cat', 'rand', 'zero'}")
    else:
        raise NotImplementedError("Entered phonological embedding code but phon_type != 'phon'")


def default_emb(emb_dim, phon_info):
    phon_type, phon_pad, ngram_size = phon_info['type'], phon_info['pad'], phon_info['gram']
    if phon_type == 'phon':
        if phon_pad == 'rand':
            return [0] * PHON_EMB_LEN * ngram_size # Is there a better way?
        elif phon_pad == 'cat':
            return [0] * emb_dim # Is there a better way?
        elif phon_pad == 'zero':
            return [0] * emb_dim
        else:
            raise NotImplementedError("Phon padding str not in {'cat', 'rand', 'zero'}")
    else:
        raise NotImplementedError("Entered phonological embedding code but phon_type != 'phon'")


def many_w2fv(wordlist, phon_info, epi_lang='hat-Latn-bab', emb_dim=512, seed=0):
    """
    """
    # seed random processes (if necessary)
    if seed:
        seed_everything(seed)
    # Set ngram size
    ngram_size = phon_info['gram']

    print("Creating phonlogical embeddings from vocab list....", flush=True)
    epi = epitran.Epitran(epi_lang)

    emb_dict = {}
    for word in wordlist:
        # First transliterate
        try:
            ipa = epi.transliterate(u''+word)
        except:
            print("Epitran WARNING:", word, flush=True)
            padded = default_emb(emb_dim, phon_info)
            emb_dict[word] = padded
            continue
        if not ipa:
            padded = default_emb(emb_dim, phon_info)
            emb_dict[word] = padded
            continue
        if epi_lang == 'hat-Latn-bab': # FIXME
            ipa = ipa.replace('ã','ɑ̃').replace('ũ','un')
        # Now obtain embedding
        if ngram_size == 1:
            try:
                vec = ft.word_to_bag_of_features(ipa)
                padded = pad_vec(vec, emb_dim, phon_info)
            except:
                print("Panphon WARNING:", ipa, flush=True)
                padded = default_emb(emb_dim, phon_info)
            emb_dict[word] = padded
        else:
            phons = ft.phonemes(ipa)
            # Case of short words with <1 ngram
            if len(phons) < ngram_size:
                vec = []
                for phon in phons:
                    try:
                        vec += ft.word_to_bag_of_features(phon)
                    except:
                        print("Panphon WARNING:", phon, '... part of ...', ipa, flush=True)
                        vec += [0] * PHON_EMB_LEN
            # Case of longer words with 1+ ngrams
            else:
                ngrams = []
                for start_i in range(len(phons) - ngram_size + 1):
                    ngram = phons[start_i:start_i + ngram_size]
                    ngrams.append(ngram)
                ngram_vecs = []
                for ngram in ngrams:
                    ngram_vec = []
                    for ngram_let in ngram:
                        try:
                            ngram_vec += ft.word_to_bag_of_features(ngram_let)
                        except:
                            print("Panphon WARNING:", ngram_let, '... part of ...', ipa, flush=True)
                            ngram_vec += [0] * PHON_EMB_LEN
                    ngram_vecs.append(ngram_vec)
                vec = list(np.sum(np.array(ngram_vecs), axis=0))
            # Now we have the vector we need
            if np.sum(np.abs(np.array(vec))) == 0:
                padded = default_emb(emb_dim, phon_info)
            else:
                padded = pad_vec(vec, emb_dim, phon_info)
            emb_dict[word] = padded
    print("Created phonological embeddings", flush=True)
    # pdb.set_trace()
    return emb_dict


def write_emb(wordlist, emb_dict, out_file):
    """
    """
    assert len(wordlist) == len(emb_dict)
    # Format embedding strings
    out_lines = []
    for word in wordlist:
        w_vec = emb_dict[word]
        line = word + ' ' + ' '.join([str(f) for f in w_vec]) + '\n'
        out_lines.append(line)
    # Write to output file
    with open(out_file, 'w') as f:
        f.writelines(out_lines)
    print("Written phon embedding to", out_file, flush=True)
    return


def make_emb_from_info(wordlist, out_file, phon_info, epi_lang='hat-Latn-bab', emb_dim=512, ngram_size=3):
    """
    """
    emb_dict = many_w2fv(wordlist, phon_info, epi_lang, emb_dim, ngram_size)
    assert len(wordlist) == len(emb_dict)
    # Format embedding strings
    out_lines = []
    for word in wordlist:
        w_vec = emb_dict[word]
        line = word + ' ' + ' '.join([str(f) for f in w_vec]) + '\n'
        out_lines.append(line)
    # Write to output file
    with open(out_file, 'w') as f:
        f.writelines(out_lines)
    print("Written phon embedding to", out_file, flush=True)
    return


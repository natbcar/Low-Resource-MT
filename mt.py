import os
import argparse
import re
from split import main as split_data
import feat_vecs as fv
import pdb

NUMOBJ = re.compile(r'[0-9]+')

LANG_REGULARIZER = {
        'ht':'hat', 'hat':'hat', 'haitian':'hat',
        'fr':'fra', 'fra':'fra', 'french':'fra',
        'en':'eng', 'eng':'eng', 'english':'eng',
        'es':'spa', 'spa':'spa', 'spanish':'spa',
        'jm':'jam', 'jam':'jam', 'jamaican':'jam',
        'th':'tha', 'tha':'tha', 'thai':'tha',
        'lo':'lao', 'lao':'lao'
        }

EPITRAN_LANGS = {
        'hat':'hat-Latn-bab',
        'fra':'fra-Latn',
        'spa':'spa-Latn',
        'jam':'jam-Latn',
        'eng':'eng-Latn',
        'tha':'tha-Thai',
        'lao':'lao-Laoo-prereform'
        }


def extract_vocab(fn):
    """
    """
    with open(fn, 'r') as f:
        lines = f.readlines()
    vocab = [line.split('\t')[0] for line in lines]
    return vocab


def create_phon_embeds(lang1_vocab_fn, lang2_vocab_fn, joint_vocab_fn, lang1, lang2, emb_fn, phon_info,\
        emb_dim, seed): # FIXME add types
    """
    """
    lang1_vocab = extract_vocab(lang1_vocab_fn)
    lang2_vocab = extract_vocab(lang2_vocab_fn)
    joint_vocab = extract_vocab(joint_vocab_fn)
    ngram_size = phon_info['gram']
    emb_dict1 = fv.many_w2fv(wordlist=lang1_vocab, phon_info=phon_info, epi_lang=EPITRAN_LANGS[lang1],\
            emb_dim=emb_dim, ngram_size=ngram_size, seed=seed)
    emb_dict2 = fv.many_w2fv(wordlist=lang2_vocab, phon_info=phon_info, epi_lang=EPITRAN_LANGS[lang2],\
            emb_dim=emb_dim, ngram_size=ngram_size, seed=seed)
    # Combine dictionaries
    for key in emb_dict1:
        emb_dict2[key] = emb_dict1[key]
    # Write embeds to file
    fv.write_emb(joint_vocab, emb_dict2, emb_fn)
    print("Embeddings written successfully to", emb_fn, flush=True)


def create_dirs(out_dir, out_dir_list=['corpora', 'pred', 'embeds', 'run', 'config']):
    """
    """
    print("Creating all out directories", flush=True)
    # main dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f"\tcreated {out_dir}", flush=True)
    # subdir's
    for dir_tail in out_dir_list:
        dir_ = os.path.join(out_dir, dir_tail)
        if not os.path.exists(dir_):
            os.mkdir(dir_)
    print(f"\tdirectories now exist for {out_dir_list}", flush=True)


def evaluate(mod_dir, src_test_data, out_path, gpu_num, tgt_test_data, mod_num='max'):
    """
    """
    # Find model path
    saved_models = os.listdir(mod_dir)
    num2saved_mod = {int(NUMOBJ.search(saved_mod)[0]):saved_mod for saved_mod in saved_models}
    mod_nums = list(num2saved_mod.keys())
    if mod_num == 'max':
        chosen_mod_num = max(mod_nums)
    else:
        chosen_mod_num = mod_num
    try:
        mod_path_tail = num2saved_mod[int(chosen_mod_num)]
    except:
        raise ValueError("Invalid model number {} with saved models {}".format(\
                chosen_mod_num, saved_models))
    mod_path = os.path.join(mod_dir, mod_path_tail)
    # Now evaluate via onmt command
    print("Evaluating model saved at", mod_path, flush=True)
    eval_cmd_temp = 'onmt_translate -model {} -src {} -output {} -gpu {}'
    bleu_cmd_temp = 'sacrebleu {} -i {} -m bleu'
    chrf_cmd_temp = 'sacrebleu {} -i {} -m chrf --chrf-word-order 2'
    # first predict translations
    eval_cmd = eval_cmd_temp.format(mod_path, src_test_data, out_path, gpu_num)
    print("\trunning command:", eval_cmd, flush=True)
    os.system(eval_cmd)
    # then score
    bleu_cmd = bleu_cmd_temp.format(tgt_test_data, out_path)
    print("\trunning command:", bleu_cmd, flush=True)
    os.system(bleu_cmd)
    # komya chrf
    print("\teval done", flush=True)


def main(args):
    """
    args:
        out_dir, phon_type, src1, tgt1, src2, tgt2, src1_lang, src2_lang, 
        tgt_lang, train1_len, train2_len, val_len, test_len, config_temp,
        embeds_file, emb_dim, gpu_num
    """
    
    # (1) Preliminary steps ------------------------------------------------
    # Useful bool for phon embeddings or no?
    phon_bool = args.phon_type != 'base'
    # info about phon type
    phon_info = {'type':args.phon_type, 'pad':args.phon_pad, 'gram':args.phon_gram}
    # path to embeddings
    embs_path = os.path.join(args.out_dir, 'embeds', args.embeds_file)
    # Regularlize language codes
    args.src1_lang = LANG_REGULARIZER[args.src1_lang.lower()]
    args.src2_lang = LANG_REGULARIZER[args.src2_lang.lower()]
    args.tgt_lang = LANG_REGULARIZER[args.tgt_lang.lower()]

    # (2) Create directories -----------------------------------------------
    create_dirs(out_dir=args.out_dir)

    # (3) Split data using split.py -----------------------------------------
    print("Splitting data", flush=True)
    '''python3 split.py --src-file path-to-hatian  --tgt-file path-to-english 
    --out-file ../OpenNMT-py/fr-ht/corpora/en-ht-15k. --lang ht --train-len 
    15000 --val-len 5000 --test-len 5000''' 
    out_f1 = os.path.join(args.out_dir, 'corpora', '{}-{}-{}'.format(args.tgt_lang, \
            args.src1_lang, args.train1_len))
    split_data(src_file=args.src1, tgt_file=args.tgt1, out_file=out_f1+'.', \
            lang=args.src1_lang, tgt_lang= args.tgt_lang, train_len=args.train1_len, \
            val_len=args.val_len, test_len=args.test_len, seed=args.seed)
    '''python3 split.py --src-file path-to-french --tgt-file path-to-english 
    --out-file ../OpenNMT-py/fr-ht/corpora/en-fr-250k. --lang fr --train-len 
    250000 --val-len 5000 --test-len 5000'''
    out_f2 = os.path.join(args.out_dir, 'corpora', '{}-{}-{}'.format(args.tgt_lang, \
            args.src2_lang, args.train2_len))
    split_data(src_file=args.src2, tgt_file=args.tgt2, out_file=out_f2+'.', \
            lang=args.src2_lang, tgt_lang=args.tgt_lang, train_len=args.train2_len, \
            val_len=args.val_len, test_len=args.test_len)

    # (4) Construct config files --------------------------------------------
    print("Opening config template", flush=True)
    with open(args.config_temp, 'r') as f:
        conf_temp_text = f.read()
    # The order of {} is: (0) data (format below), (1) phon_type
    #   and replace OUTDIR with args.out_dir
    #   and VOCAB with vocab type
    #   and EMBINFO for embeddings info
    #   and WORDVECSIZE for emb dim
    #   and SAVESTEPS for save checkpoint steps (default 30000)
    #   and TRAINSTEPS for training steps (default 60000)
    #   and VALSTEPS for validation steps (default 5000)
    # FIXME and gpu num ?
    print("Template should have {} for the data paragraph, {} for the phon_type, "
          "'OUTDIR' for output directory, and VOCAB for vocab type", flush=True)
    # data strings
    full_data_str = f'''data:
    corpus_1:
        path_src: {out_f1}.{args.src1_lang}.train
        path_tgt: {out_f1}.{args.tgt_lang}.train
    corpus_2:
        path_src: {out_f2}.{args.src2_lang}.train
        path_tgt: {out_f2}.{args.tgt_lang}.train
    valid:
        path_src: {out_f1}.{args.src1_lang}.val
        path_tgt: {out_f1}.{args.tgt_lang}.val'''
    lang1_data_str = f'''data:
    corpus_1:
        path_src: {out_f1}.{args.src1_lang}.train
        path_tgt: {out_f1}.{args.tgt_lang}.train
    valid:
        path_src: {out_f1}.{args.src1_lang}.val
        path_tgt: {out_f1}.{args.tgt_lang}.val'''
    lang2_data_str = f'''data:
    corpus_2:
        path_src: {out_f2}.{args.src2_lang}.train
        path_tgt: {out_f2}.{args.tgt_lang}.train'''
    # embedding string
    emb_info = f'''src_embeddings: {embs_path}
embeddings_type: "GloVe"'''
    # fill templates
    mostly_filled_temp = conf_temp_text.replace('OUTDIR', args.out_dir).replace(\
            'EMBINFO', emb_info).replace('WORDVECSIZE', str(args.emb_dim)).replace(\
            'SAVESTEPS', str(args.save_every)).replace('TRAINSTEPS', str(args.train_steps)).replace(\
            'VALSTEPS', str(args.val_steps))
    full_conf_text = mostly_filled_temp.replace('VOCAB', '').format(full_data_str,\
            args.phon_type)
    lang1_conf_text = mostly_filled_temp.replace('VOCAB', '-'+args.src1_lang.upper()).format(\
            lang1_data_str, args.phon_type)
    lang2_conf_text = mostly_filled_temp.replace('VOCAB', '-'+args.src2_lang.upper()).format(\
            lang2_data_str, args.phon_type)
    # write yaml files
    # example: fr-ht/config/fr-ht-phon.yaml
    full_conf_fn, lang1_conf_fn, lang2_conf_fn = \
      f'{args.out_dir}/config/{args.out_dir}-{args.phon_type}.yaml',\
      f'{args.out_dir}/config/{args.out_dir}-{args.phon_type}-{args.src1_lang.upper()}.yaml',\
      f'{args.out_dir}/config/{args.out_dir}-{args.phon_type}-{args.src2_lang.upper()}.yaml'
    # Write config files
    with open(full_conf_fn, 'w') as f:
        f.write(full_conf_text)
    if phon_bool:
        with open(lang1_conf_fn, 'w') as f:
            f.write(lang1_conf_text)
        with open(lang2_conf_fn, 'w') as f:
            f.write(lang2_conf_text)
        print("Config *.yaml files written to {}, {}, {}".format(full_conf_fn, \
                lang1_conf_fn, lang2_conf_fn), flush=True)
    else:
        print(f"Config *.yaml file written to {full_conf_fn}", flush=True)

    # (5) Construct vocab ---------------------------------------------------
    vocab_cmd_temp = "onmt_build_vocab -config {} -n_sample -1"
    # joint vocab
    joint_vocab_cmd = vocab_cmd_temp.format(full_conf_fn)
    print("Executing:", joint_vocab_cmd, flush=True)
    os.system(joint_vocab_cmd)
    if phon_bool:
        # lang1 vocab
        lang1_vocab_cmd = vocab_cmd_temp.format(lang1_conf_fn)
        print("Executing:", lang1_vocab_cmd, flush=True)
        os.system(lang1_vocab_cmd)
        # lang2 vocab
        lang2_vocab_cmd = vocab_cmd_temp.format(lang2_conf_fn)
        print("Executing:", lang2_vocab_cmd, flush=True)
        os.system(lang2_vocab_cmd)

    # (6) Create embeddings -------------------------------------------------
    vocab_fn_template = args.out_dir + '/run/src{}.vocab'
    lang1_vocab_fn = vocab_fn_template.format('-'+args.src1_lang.upper())
    lang2_vocab_fn = vocab_fn_template.format('-'+args.src2_lang.upper())
    joint_vocab_fn = vocab_fn_template.format('')
    if phon_bool:
        create_phon_embeds(lang1_vocab_fn, lang2_vocab_fn, joint_vocab_fn, args.src1_lang, \
            args.src2_lang, embs_path, phon_info, args.emb_dim, args.seed)

    # (7) Train model -------------------------------------------------------
    print("Training MT model....", flush=True)
    train_cmd = f'onmt_train -config {full_conf_fn}'
    print('\trunning', train_cmd, flush=True)
    os.system(train_cmd)
    
    # (8) Evaluate and score ------------------------------------------------
    # model dir to find model path
    mod_dir = os.path.join(args.out_dir, 'run', args.phon_type)
    # testing data for src1 only
    src_test_data = f'{out_f1}.{args.src1_lang}.test'
    tgt_test_data = f'{out_f1}.{args.tgt_lang}.test'
    # output file for predictions
    out_pred_tail = '{}-preds-{}.txt'.format(args.phon_type, args.test_len)
    out_pred_file = os.path.join(args.out_dir, 'pred', out_pred_tail)
    # Now we can evaluate
    evaluate(mod_dir=mod_dir, src_test_data=src_test_data, out_path=out_pred_file, gpu_num=args.gpu_num, tgt_test_data=tgt_test_data, mod_num=args.model_eval_num)
    
    return


if __name__=='__main__':
    """
    args:
        out_dir, phon_type, src1, tgt1, src2, tgt2, src1_lang, src2_lang,
        tgt_lang, train1_len, train2_len, val_len, test_len, config_temp
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--out-dir', type=str, 
            help='main dir where everything will be written',
            required=True)
    parser.add_argument('--phon-type', type=str,
            help='Use phonological embeddings? Choose "base" for no phon embeds or "phon"',
            choices=['base', 'phon'],
            required=True)
    parser.add_argument('--phon-pad', type=str,
            help='How to pad phonological embeddings?',
            choices=['cat', 'rand', 'zero'],
            default='random')
    parser.add_argument('--phon-gram', type=int,
            help='Organize phon embeddings via ngrams or unigrams? '
            'For unigrams, set to 1; for ngrams, set to n',
            default=3)
    parser.add_argument('--src1', type=str,
            help='Bi-text file for first source language (LRL, testing language)',
            required=True)
    parser.add_argument('--src2', type=str,
            help='Bi-text file for second source language (HRL, transfer language)',
            required=True)
    parser.add_argument('--tgt1', type=str,
            help='Target language (TGT) file aligned with src1',
            required=True)
    parser.add_argument('--tgt2', type=str,
            help='Target language (TGT) file aligned with src2',
            required=True)
    parser.add_argument('--src1_lang', type=str,
            help='First source language (LRL, testing language)',
            required=True)
    parser.add_argument('--src2_lang', type=str,
            help='Second source language (HRL, transfer language)',
            required=True)
    parser.add_argument('--tgt-lang', type=str,
            help='Target langauge (TGT)',
            default='en')
    parser.add_argument('--train1-len', type=int,
            help='Length of training data for first source language (LRL)',
            default=15000)
    parser.add_argument('--train2-len', type=int,
            help='Length of training data for second source language (HRL)',
            default=250000)
    parser.add_argument('--val-len', type=int,
            help='Length of val data, which is in the LRL or testing language',
            default=5000)
    parser.add_argument('--test-len', type=int,
            help='Length of testing data, which is in the LRL or testing language',
            default=5000)
    parser.add_argument('--config-temp', type=str,
            help='File for config template',
            default='config_template')
    parser.add_argument('--embeds-file', type=str,
            help='embeddings file name',
            default='phon_embeddings')
    parser.add_argument('--emb-dim', type=int,
            help='model dimension',
            default=512)
    parser.add_argument('--gpu-num', type=int,
            help='which GPU to use',
            default=0)
    parser.add_argument('--train-steps', type=int,
            help='How many training steps',
            default=60000)
    parser.add_argument('--val-steps', type=int,
            help='Validation steps (how often to validate)',
            default=5000)
    parser.add_argument('--save-every', type=int,
            help='Save the model every how many steps',
            default=30000)
    parser.add_argument('--model-eval-num', default='max',
            help='which model to test')
    parser.add_argument("--seed", type=int,
            default=sum(bytes(b'dragn')),
            help="random seed, set to 0 for no seed")

    args = parser.parse_args()

    main(args)

    '''Example usage:
    python3 mt.py --out-dir test-test --phon-type phon --phon-pad rand --phon-gram 3 --src1 enht_haitian --tgt1 enht_english --src2 enfr_french --tgt2 enfr_english --src1_lang ht --src2_lang fr --tgt-lang en --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --emb-dim 512
    '''
    '''Or on patient:
    python3 mt.py --out-dir test-test --phon-type phon --phon-pad rand --phon-gram 3 --src1 ../translation/enht_haitian --tgt1 ../translation/enht_english --src2 ../translation/enfr_french --tgt2 ../translation/enfr_english --src1_lang ht --src2_lang fr --tgt-lang en --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --emb-dim 512 --train-steps 1000 --save-every 1000 --val-steps 250
    '''

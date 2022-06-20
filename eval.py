import argparse
from mt import evaluate

if __name__=='__main__':
    '''def evaluate(mod_dir: str, src_test_data: str, out_path: str, gpu_num: int, tgt_test_data: str,\
        mod_num='max', decode_mod=None)
    Example call:
    evaluate(mod_dir=mod_dir, src_test_data=src_test_data, out_path=out_pred_file,\
            gpu_num=args.gpu_num, tgt_test_data=tgt_test_data, mod_num=args.model_eval_num,\
            decode_mod=eval_decode_mod)
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--mod-dir", type=str,
            help="Directory where models are saved",
            required=True)
    parser.add_argument("--src-test-data", type=str,
            help="Path to source side of test bitext",
            required=True)
    parser.add_argument("--tgt-test-data", type=str,
            help="Path to target side of text bitext",
            required=True)
    parser.add_argument("--pred-output", type=str,
            help="Path to file to write predicted translations to",
            required=True)
    parser.add_argument("--gpu-num", type=int,
            help="Which GPU to use?",
            default=0)
    parser.add_argument("--mod-num", default='max',
            help="Which model to evaluate?")
    parser.add_argument("--tgt-spm-mod", default=None,
            help="Path to TGT language sentencepiece decoding model")

    args = parser.parse_args()

    evaluate(mod_dir=args.mod_dir, src_test_data=args.src_test_data, out_path=args.pred_output,\
            gpu_num=args.gpu_num, tgt_test_data=args.tgt_test_data, mod_num=args.mod_num,\
            decode_mod=args.tgt_spm_mod)

    '''
    python3 eval.py --mod-dir /usr1/data/nrrobins/translation/onmt_outputs/ --src-test-data /usr1/data/nrrobins/translation/onmt_outputs/fr-ht/corpora/eng-hat-15000.hat.test --tgt-test-data /usr1/data/nrrobins/translation/onmt_outputs/fr-ht/corpora/eng-hat-15000.eng.test --pred-output /usr1/data/nrrobins/translation/onmt_outputs/fr-ht/pred/base-preds-5000-ep30000.txt --mod-num 30000
    '''

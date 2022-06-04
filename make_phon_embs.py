import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-path",
                        type=str,
                        help="path to pkl file for phon embs")
    parser.add_argument("--out-path",
                        type=str,
                        help="path to save formatted phon embs to")
    parser.add_argument("--cat",
                        type=int,
                        default=0,
                        help="whether or not to concatenate phon embs together to match model dimension, assumes it is 512.")
    args = parser.parse_args()

    with open(args.emb_path, "rb") as f:
        emb_dict = pkl.load(f)
    
    if args.cat:
        key = next(iter(emb_dict))
        emb_len = len(emb_dict[key])
        n_cat = 512 // emb_len
        n_pad = 512 % emb_len
        concat_emb_dict = dict()
        for k in emb_dict.keys():
            concat_emb = np.concatenate([emb_dict[k] for _ in range(n_cat)])
            concat_emb = np.pad(concat_emb, (0, n_pad), "wrap")
            assert len(concat_emb) == 512
            concat_emb_dict[k] = concat_emb
        
        for k, v in concat_emb_dict.items():
            emb_str = " ".join([k.decode("utf-8")] + [str(float(digit)) for digit in v])
            with open(outfile, "a") as f:
                f.write(emb_str+"\n")
        
    else:
        for k, v in emb_dict.items():
            emb_str = " ".join([k.decode("utf-8")] + [str(float(digit)) for digit in v])
            with open(args.out_path, "w") as f:
                f.write(emb_str+"\n")
            

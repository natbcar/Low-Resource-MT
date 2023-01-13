CUDA_VISIBLE_DEVICES=0
FRHTDATA=/usr1/data/nrrobins/translation/onmt_corpora/fra_hat
FRHTOUT=/usr1/data/nrrobins/translation/onmt_outputs/fr-ht
# files: enfr_english  enfr_french  enht_english  enht_haitian
echo "G2P XPER'T %%%%%%"
python3 mt.py --out-dir $FRHTOUT/g2pbpe --phon-type base --phon-pad rand --phon-gram 1 --src1 $FRHTDATA/enht_haitian --tgt1 $FRHTDATA/enht_english --src2 $FRHTDATA/enfr_french --tgt2 $FRHTDATA/enfr_english --src1-lang hat --src2-lang fra --tgt-lang eng --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --mod-dim 512 --train-steps 200000 --save-steps 50000 --val-steps 5000 --spm-vocab-size 8000

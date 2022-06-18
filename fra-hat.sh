CUDA_VISIBLE_DEVICES=3
FRAHAT=/usr1/data/nrrobins/translation/onmt_corpora/fra_hat/
# files: enfr_english  enfr_french  enht_english  enht_haitian
echo "BASE XPER'T %%%%%%"
python3 mt.py --out-dir fr-ht --phon-type base --phon-pad zero --phon-gram 1 --src1 $FRAHAT/enht_haitian --tgt1 $FRAHAT/enht_english --src2 $FRAHAT/enfr_french --tgt2 $FRAHAT/enfr_english --src1_lang hat --src2_lang fra --tgt-lang eng --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --spm-vocab-size 8000
echo "PHON RAND NGRAM XPER'T %%%%%%"
python3 mt.py --out-dir fr-ht --phon-type phon --phon-pad rand --phon-gram 3 --src1 $FRAHAT/enht_haitian --tgt1 $FRAHAT/enht_english --src2 $FRAHAT/enfr_french --tgt2 $FRAHAT/enfr_english --src1_lang hat --src2_lang fra --tgt-lang eng --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --spm-vocab-size 8000
echo "PHON RAND 1GRAM XPER'T %%%%%%"
python3 mt.py --out-dir fr-ht --phon-type phon --phon-pad rand --phon-gram 1 --src1 $FRAHAT/enht_haitian --tgt1 $FRAHAT/enht_english --src2 $FRAHAT/enfr_french --tgt2 $FRAHAT/enfr_english --src1_lang hat --src2_lang fra --tgt-lang eng --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --spm-vocab-size 8000
echo "PHON ZERO NGRAM XPER'T %%%%%%"
python3 mt.py --out-dir fr-ht --phon-type phon --phon-pad zero --phon-gram 3 --src1 $FRAHAT/enht_haitian --tgt1 $FRAHAT/enht_english --src2 $FRAHAT/enfr_french --tgt2 $FRAHAT/enfr_english --src1_lang hat --src2_lang fra --tgt-lang eng --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --spm-vocab-size 8000

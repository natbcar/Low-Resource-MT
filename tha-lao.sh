THALAO=/usr1/data/nrrobins/translation/onmt_corpora/tha_lao
# files: lao-eng.english  lao-eng.lao  tha-eng.english  tha-eng.thai
echo "BASE XPER'T %%%%%%"
python3 mt.py --out-dir th-lo --phon-type base --phon-pad zero --phon-gram 1 --src1 $THALAO/lao-eng.lao --tgt1 $THALAO/lao-eng.english --src2 $THALAO/tha-eng.thai --tgt2 $THALAO/tha-eng.english --src1_lang lao --src2_lang tha --tgt-lang eng --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --use-spm
echo "PHON RAND NGRAM XPER'T %%%%%%"
python3 mt.py --out-dir th-lo --phon-type phon --phon-pad rand --phon-gram 3 --src1 $THALAO/lao-eng.lao --tgt1 $THALAO/lao-eng.english --src2 $THALAO/tha-eng.thai --tgt2 $THALAO/tha-eng.english --src1_lang lao --src2_lang tha --tgt-lang eng --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --use-spm
echo "PHON RAND 1GRAM XPER'T %%%%%%"
python3 mt.py --out-dir th-lo --phon-type phon --phon-pad rand --phon-gram 1 --src1 $THALAO/lao-eng.lao --tgt1 $THALAO/lao-eng.english --src2 $THALAO/tha-eng.thai --tgt2 $THALAO/tha-eng.english --src1_lang lao --src2_lang tha --tgt-lang eng --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --use-spm
echo "PHON ZERO NGRAM XPER'T %%%%%%"
python3 mt.py --out-dir th-lo --phon-type phon --phon-pad zero --phon-gram 3 --src1 $THALAO/lao-eng.lao --tgt1 $THALAO/lao-eng.english --src2 $THALAO/tha-eng.thai --tgt2 $THALAO/tha-eng.english --src1_lang lao --src2_lang tha --tgt-lang eng --train1-len 15000 --train2-len 250000 --val-len 5000 --test-len 5000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --use-spm

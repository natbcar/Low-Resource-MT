ENJMDATA=/usr1/data/nrrobins/translation/onmt_corpora/eng_jam
ENJMOUT=/usr1/data/nrrobins/translation/onmt_outputs/en-jm
# files: eng-fra.english  eng-fra.french  jam-fra.french  jam-fra.jamaican
echo "BASE XPER'T %%%%%%"
python3 mt.py --out-dir $ENJMOUT --phon-type base --phon-pad zero --phon-gram 1 --src1 $ENJMDATA/jam-fra.jamaican --tgt1 $ENJMDATA/jam-fra.french --src2 $ENJMDATA/eng-fra.english --tgt2 $ENJMDATA/eng-fra.french --src1-lang jam --src2-lang eng --tgt-lang fra --train1-len 5925 --train2-len 250000 --val-len 1000 --test-len 1000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --spm-vocab-size 8000
echo "PHON RAND NGRAM XPER'T %%%%%%"
python3 mt.py --out-dir $ENJMOUT --phon-type phon --phon-pad rand --phon-gram 3 --src1 $ENJMDATA/jam-fra.jamaican --tgt1 $ENJMDATA/jam-fra.french --src2 $ENJMDATA/eng-fra.english --tgt2 $ENJMDATA/eng-fra.french --src1-lang jam --src2-lang eng --tgt-lang fra --train1-len 5925 --train2-len 250000 --val-len 1000 --test-len 1000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --spm-vocab-size 8000
echo "PHON RAND 1GRAM XPER'T %%%%%%"
python3 mt.py --out-dir $ENJMOUT --phon-type phon --phon-pad rand --phon-gram 1 --src1 $ENJMDATA/jam-fra.jamaican --tgt1 $ENJMDATA/jam-fra.french --src2 $ENJMDATA/eng-fra.english --tgt2 $ENJMDATA/eng-fra.french --src1-lang jam --src2-lang eng --tgt-lang fra --train1-len 5925 --train2-len 250000 --val-len 1000 --test-len 1000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --spm-vocab-size 8000
echo "PHON ZERO NGRAM XPER'T %%%%%%"
python3 mt.py --out-dir $ENJMOUT --phon-type phon --phon-pad zero --phon-gram 3 --src1 $ENJMDATA/jam-fra.jamaican --tgt1 $ENJMDATA/jam-fra.french --src2 $ENJMDATA/eng-fra.english --tgt2 $ENJMDATA/eng-fra.french --src1-lang jam --src2-lang eng --tgt-lang fra --train1-len 5925 --train2-len 250000 --val-len 1000 --test-len 1000 --config-temp config_template --mod-dim 512 --train-steps 60000 --save-steps 30000 --val-steps 5000 --spm-vocab-size 8000

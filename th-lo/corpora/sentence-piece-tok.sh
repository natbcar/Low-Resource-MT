% Lao
spm_train --input=en-lo.lo.train --model_prefix=lo-smp --vocab_size=16000 --character_coverage=1.0 --model_type=bpe
spm_encode --model=lo-smp.model < en-lo.lo.train > en-lo.lo.train.sp
spm_encode --model=lo-smp.model < en-lo.lo.val > en-lo.lo.val.sp
spm_encode --model=lo-smp.model < en-lo.lo.test > en-lo.lo.test.sp

% Thai
spm_train --input=en-th.th.train --model_prefix=th-smp --vocab_size=16000 --character_coverage=1.0 --model_type=bpe
spm_encode --model=th-smp.model < en-th.th.train > en-th.th.train.sp
spm_encode --model=th-smp.model < en-th.th.val > en-th.th.val.sp
spm_encode --model=th-smp.model < en-th.th.test > en-th.th.test.sp

% English
spm_train --input=en.train.complete --model_prefix=en-smp --vocab_size=16000 --character_coverage=1.0 --model_type=bpe
spm_encode --model=en-smp.model < en-lo.en.train > en-lo.en.train.sp
spm_encode --model=en-smp.model < en-th.en.train > en-th.en.train.sp
spm_encode --model=en-smp.model < en-lo.en.val > en-lo.en.val.sp
spm_encode --model=en-smp.model < en-th.en.val > en-th.en.val.sp
spm_encode --model=en-smp.model < en-lo.en.test > en-lo.en.test.sp
spm_encode --model=en-smp.model < en-th.en.test > en-th.en.test.sp

Decoding predictions
% spm_decode --model=en-smp.model --input_format=piece < ../preds/base-sp.txt > ../preds/base-sp-detok.txt
% spm_decode --model=th-lo/corpora/en-smp.model --input_format=piece < th-lo/preds/phon-sp.txt > th-lo/preds/phon-sp-detok.txt

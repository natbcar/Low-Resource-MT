#!/bin/bash
[ $# -eq 0 ] && { echo "Usage: $0 <save dir>"; exit 1; }
DATADIR=$1
# Make directories
mkdir -p $DATADIR/fra_hat $DATADIR/tur_aze $DATADIR/tha_lao $DATADIR/eng_jam
# Download hat-eng bitext and fra-eng bitext
wget -O $DATADIR/fra_hat/enht_haitian https://raw.githubusercontent.com/n8rob/corpora/master/french_haitian/ht_total.txt
wget -O $DATADIR/fra_hat/enht_english https://raw.githubusercontent.com/n8rob/corpora/master/french_haitian/en_total.txt
wget -O $DATADIR/fra_hat/enfr_french https://raw.githubusercontent.com/n8rob/corpora/master/french_haitian/enfr_fr_fromchurch.txt
wget -O $DATADIR/fra_hat/enfr_english https://raw.githubusercontent.com/n8rob/corpora/master/french_haitian/enfr_en_fromchurch.txt
# Download jam-fra bitext and eng-fra bitext
wget -O $DATADIR/eng_jam/jam-fra.jamaican https://raw.githubusercontent.com/n8rob/corpora/master/english_jamaican/jm_jamfra_src.txt
wget -O $DATADIR/eng_jam/jam-fra.french https://raw.githubusercontent.com/n8rob/corpora/master/english_jamaican/fr_jamfra_tgt.txt
wget -O $DATADIR/eng_jam/eng-fra.english https://raw.githubusercontent.com/n8rob/corpora/master/english_jamaican/enfr_english
wget -O $DATADIR/eng_jam/eng-fra.french https://raw.githubusercontent.com/n8rob/corpora/master/english_jamaican/enfr_french
# Download tha-eng bitext and lao-eng bitext
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RDYH391r7xxKBuUA5pnuaLqyYM5BmzX5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RDYH391r7xxKBuUA5pnuaLqyYM5BmzX5" -O $DATADIR/tha_lao/tha-eng.thai && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hcZIS2apFsSwkB5X_x_27UtYE3f6ljZK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hcZIS2apFsSwkB5X_x_27UtYE3f6ljZK" -O $DATADIR/tha_lao/tha-eng.english && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n7ZxQnR834_RkDyO8EAYm2b8ITVMy-QX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n7ZxQnR834_RkDyO8EAYm2b8ITVMy-QX" -O $DATADIR/tha_lao/lao-eng.lao && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SdlfOYQ0W6gWJKw_SCyo2zjhEltcrrPV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1SdlfOYQ0W6gWJKw_SCyo2zjhEltcrrPV" -O $DATADIR/tha_lao/lao-eng.english && rm -rf /tmp/cookies.txt
# Download eng-tur bitext and eng-aze bitext
gsutil -m cp -r gs://til-corpus/corpus/train/tr-en/tr-en.en $DATADIR/tur_aze/tur-eng.english
gsutil -m cp -r gs://til-corpus/corpus/train/tr-en/tr-en.tr $DATADIR/tur_aze/tur-eng.turkish
#python3 clean_tr.py $DATADIR/tur_aze/tur-eng.turkish $DATADIR/tur_aze/tur-eng.english
gsutil -m cp -r gs://til-corpus/corpus/train/az-en/az-en.en $DATADIR/tur_aze/aze-eng.english
gsutil -m cp -r gs://til-corpus/corpus/train/az-en/az-en.az $DATADIR/tur_aze/aze-eng.azerbaijani
#python3 clean_tr.py $DATADIR/tur_aze/aze-eng.azerbaijani $DATADIR/tur_aze/aze-eng.english

#!/bin/bash

# This script downloads all necessary monolingual data, and trims them to the amount used in the experiments

cd ..
. ./local-settings.sh 
cd data

STD_DATA=newscrawl-mono
mkdir -p $STD_DATA

#################
# Downloading newscrawl data
cd $STD_DATA
LANGS="uk ta bn en"
echo "Download WMT monolingual data"
for lang in $LANGS; do
	mkdir -p $lang
	cd $lang
	wget -c --no-check-certificate https://data.statmt.org/news-crawl/$lang/news.2020.$lang.shuffled.deduped.gz
	gunzip news.2020.$lang.shuffled.deduped.gz
	cd ..
done
cd ..

#################
# Germanic & Romance Languages
for lang in en uk
do

	head -1000000 $STD_DATA/$lang/news.2020.$lang.shuffled.deduped \
		> news.2020.$lang.1M

	for size in 1M
       	do
		if [ $lang == uk ]; then
			tok_lang=ru
		else
			tok_lang=$lang
		fi
		cat news.2020.$lang.$size \
			| $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l $tok_lang \
			| $MOSES_SCRIPTS/tokenizer/lowercase.perl \
			| $MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl \
			| $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l $tok_lang \
				-threads 12 > news.2020.$lang.tok.$size
	done
done

#################
# Downloading full English data
cd $STD_DATA/en
for year in 2018 2019; do 
	wget -c --no-check-certificate https://data.statmt.org/news-crawl/en/news.$year.en.shuffled.deduped.gz
	gunzip news.$year.en.shuffled.deduped.gz
done

cat news.2018.en.shuffled.deduped news.2019.en.shuffled.deduped news.2020.en.shuffled.deduped > news.2018-20.en.full
cd ../..

# Full English - Added 6 June 2022.
cat $STD_DATA/en/news.2018-20.en.full \
	| $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l en \
	| $MOSES_SCRIPTS/tokenizer/lowercase.perl \
	| $MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl \
	| $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l en \
		-threads 12 > news.2018-20.en.tok.full


#################
# Downloading commoncrawl data
wget -c https://data.statmt.org/wmt19/parallel-corpus-filtering/commoncrawl.deduped.en.xz

CC_DATA=commoncrawl.deduped.en.xz
# English Common Crawl
OUTFILE=commoncrawl.en.tok
xzcat $CC_DATA | head -100000000 \
	| $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l en \
	| $MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl \
	| sed 's/\xC2\xA0/ /g' \
	| $MOSES_SCRIPTS/tokenizer/lowercase.perl \
	| $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l en \
		-threads 12 > $OUTFILE

################
# Indic Languages.
# Downloading tokenizer script from m2m100:

wget https://raw.githubusercontent.com/facebookresearch/fairseq/main/examples/m2m_100/tokenizers/tokenize_indic.py

for lang in ta bn
do

	head -1000000 $STD_DATA/$lang/news.2020.$lang.shuffled.deduped \
		> news.2020.$lang.1M

	for size in 1M
	do
		cat news.2020.$lang.$size | python3 tokenize_indic.py \
			$lang > news.2020.$lang.tok.$size
	done
done

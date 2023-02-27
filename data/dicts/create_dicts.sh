HOST=https://dl.fbaipublicfiles.com/arrival/dictionaries

echo Collecting bilingual dictionaries... 
for pair in bn-en ta-en uk-en 
do
	mkdir -p $pair/dev $pair/train $pair/test
	wget $HOST/$pair.txt -P $pair
	wget $HOST/$pair.0-5000.txt -P $pair/train
	wget $HOST/$pair.5000-6500.txt -P $pair/test
done

wget https://raw.githubusercontent.com/kellymarchisio/euc-v-graph-bli/main/dicts/one-to-one.py

for src in bn ta uk
do
    echo Making dev sets...
    python make_devsets.py $src en
    echo Making data one-to-one...
    python one-to-one.py $src-en/train/$src-en.0-5000.txt 2
done
    
echo Dictionary creation done.

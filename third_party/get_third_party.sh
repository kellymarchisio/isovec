#!/bin/bash

git clone https://github.com/artetxem/vecmap.git
mv vecmap vecmap_fork

cd vecmap_fork
wget https://raw.githubusercontent.com/kellymarchisio/align-semisup-bli/main/scripts/map_embeddings.py -O map_embeddings.py

cd ..
git clone https://github.com/tmikolov/word2vec.git
cd word2vec && make && cd ..
# Convert vectors to bin format for monolingual eval
git clone https://github.com/marekrei/convertvec.git




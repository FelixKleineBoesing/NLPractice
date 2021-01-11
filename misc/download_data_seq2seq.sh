#!/bin/bash

if ! [ -d "./data" ]; then
  mkdir ./data
fi
if ! [ -d "./data/german-english" ]; then
  mkdir ./data/german-english
fi

curl https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de -o ./data/german-english/german.txt
curl https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en -o ./data/german-english/english.txt


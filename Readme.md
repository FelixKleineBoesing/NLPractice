# NLP Practice

#####  Implementation of NLP Models with numpy and scipy

### Installation

...

### Content

The following Algorithms are implemented in this repo: 

Primary:
- Word2Vec
- seq2seq

Secondary:
- SGD
- Adam
- RMSProp

- BeamSearchEncoding

### Future Content

Primary:

-

Secondary:
- Huffmann Encoding for Word2Vec
- Negative Sampling for Word2vec



### Datasets

##### Seq2Seq

Run the bash file download_data_seq2seq.sh in root dir to create the directories and 
download the german and english dataset.
```
sh ./misc/download_data_seq2seq.sh
```

If you want to do it manually, execute the following steps:

Download the german and english dataset by [Stanford NLP Group](https://nlp.stanford.edu/projects/nmt/):
- German Dataset (https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de)
- English Dataset (https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en)

Save them as english.txt and german.txt in the directory data/german-english/*



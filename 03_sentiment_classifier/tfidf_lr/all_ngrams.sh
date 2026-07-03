#!/bin/bash

python3 train_tfidf.py

python3 train_tfidf.py --ngrams unigrams

python3 train_tfidf.py --ngrams bigrams

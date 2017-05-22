library(wordVectors)




if (!file.exists("cookbooks.zip")) {
  download.file("http://archive.lib.msu.edu/dinfo/feedingamerica/cookbook_text.zip","cookbooks.zip")
}

unzip("cookbooks.zip",exdir="cookbooks")


sents <- readLines("input.txt")

library(stringr)
library(stringi)

sentss <- stri_replace_all_regex(stri_trans_nfkc(sents), pattern = '[:blank:]', replacement = '')

sentsss <- paste(str_split(sentss, ''), collapse = ' ')
sentsss[



model = train_word2vec("cookbooks.txt","cookbook_vectors.bin",
                       vectors=200,threads=4,window=12,iter=5,negative_samples=0)





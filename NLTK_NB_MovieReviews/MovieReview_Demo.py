# -- Naive Bayes Classification Code from Chapter 6 of 
# -- Natural Language Processing with Python, by Steven Bird, Ewan Klein, and Edward Loper.

import nltk
from nltk.corpus import movie_reviews
import random

# - categories 'neg' and 'pos'
# - fileid ex - 'neg/cv000_29416.txt'
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# -- A list of the 2,000 most frequent words in the overall corpus 
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()[:2000] # [_document-classify-all-words]

# -- Define a Feature Extractor
def document_features(document): 
    document_words = set(document) # compute the set of all words in a document
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

# -- Calling print document_features(movie_reviews.words('pos/cv957_8737.txt'))
# -- Would produce - {'contains(waste)': False, 'contains(lot)': False, ...}

# -- for brevity, we are not creating a dev-test set here
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print "\nClassifier Accuracy : %4f\n" % nltk.classify.accuracy(classifier, test_set) 

#uncomment this line to show the model's 10 most informative features
#classifier.show_most_informative_features(10) 

# -- Current Movie Reviews -- Obtain from rogertebert.com
#---------------------------------------------------------
# -- Atlas Shrugged     1 / 5 stars
# -- Kung Fu Panda 2  3.5 / 5 stars
# -- Tree of Life       4 / 5 stars

movie_file_names = [
                    "AtlasShrugged_Review.txt",
                    "KungFuPanda2_Review.txt",
                    "TreeOfLife_Review.txt"
                   ]
for movie_file_name in movie_file_names:
	my_movie = open("%s" % (movie_file_name) , "r")
	text = my_movie.read()
	probs = classifier.prob_classify(document_features(nltk.word_tokenize(text)))
	my_movie.close()
	print "\n%s's probability dist ---> neg - %.4f  pos - %.4f\n" % \
			(movie_file_name, probs.prob('neg'), probs.prob('pos'))

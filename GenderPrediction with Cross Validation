from nltk.corpus import names
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import random
import nltk
nltk.download('names')

# https://www.nltk.org/book/ch06.html

def gender_features(word):
    return {'last_letter': word[-1]}
print(gender_features('Shrek'))


def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

def gender_features3(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
    [(name, 'female') for name in names.words('female.txt')])

random.shuffle(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

featuresets2 = [(gender_features2(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets2[500:], featuresets2[:500]
classifier2 = nltk.NaiveBayesClassifier.train(train_set)

featuresets3 = [(gender_features3(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets3[500:], featuresets2[:500]
classifier3 = nltk.NaiveBayesClassifier.train(train_set)

# Scoring
print(classifier.classify(gender_features2('Neo')))

print(classifier.classify(gender_features2('Trinity')))

print(classifier.classify(gender_features('Tony')))
type(classifier)

cv = KFold(n_splits=10)
for traincv, testcv in cv.split(train_set):
    classifier = nltk.NaiveBayesClassifier.train(train_set[traincv[0]:traincv[len(traincv)-1]])
    print ('accuracy:', nltk.classify.util.accuracy(classifier, train_set[testcv[0]:testcv[len(testcv)-1]]))

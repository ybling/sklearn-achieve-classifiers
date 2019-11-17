import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from numpy import *
from nltk.corpus import stopwords   #remove stopword
from nltk.tokenize import word_tokenize #segmentation(split word)
from sklearn.metrics import confusion_matrix
from numpy.core.umath_tests import inner1d


#========Data preprocessing=========#
def tokenization(pos_path, neg_path):
    labels=[]
    data=[]
    cachedStopWords = stopwords.words("english")
    f = open("train.txt", "a+") #"a+" continue writing
    with open(pos_path, "r", encoding='utf-8', errors='ignore')as file:# reading positive reviews and puting label 1.
        for line in file:
            labels.append("1")
            words = [word for word in word_tokenize(line) if (str.isalpha(word) is not False)]

            word_stopped = [w.lower() for w in words if
                            (w.lower() not in cachedStopWords and len(w) > 2 and str.isalpha(w) is not False)]
            # lower case, remove stopwords, remove words which length <3 and remove words contain digitals and words, like 2-year.
            data.append(word_stopped)
            f.write("1"+ ' '+ ' '.join(word_stopped) + '\n')

    with open(neg_path, "r", encoding="utf-8", errors='ignore')as file:# reading negative reviews and puting label 0.
        for line in file:
            labels.append("0")
            words = [word for word in word_tokenize(line) if (str.isalpha(word) is not False)]# segmentation
            word_stopped = [w.lower() for w in words if
                            (w.lower() not in cachedStopWords and len(w) > 2 and str.isalpha(w) is not False)]
            # lower case, remove stopwords, remove words which length <3 and remove words contain digitals and words, like 2-year.
            data.append(word_stopped)
            f.write("0" + ' ' + ' '.join(word_stopped) + '\n')
    f.close()
    return data,labels


# baseline random
def guess(test_ture):
    doc_class_predicted=[]
    report=[]
    for i in range(len(test_ture)):
        a=random.randint(0,2)
        if a == test_ture[i]:
            temp = True
            report.append(temp)
        else:
            temp = False
            report.append(temp)
        doc_class_predicted.append(a)
    report=np.array(report)
    Precision1(doc_class_predicted,report)


# ========SVM========#
def SvmClass(x_train, y_train):
    from sklearn.svm import SVC
    # call classifier
    clf = SVC(kernel='linear', probability=True)  # default with 'rbf' linear kernel
    clf.fit(x_train, y_train)  # training. For supervised learning, it is fit(X,y). For unsupervised learning, it is fit(X)
    return clf


# =====Navie Bayes=========#
def NbClass(x_train, y_train):
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=0.01).fit(x_train, y_train)
    return clf


# ========Logistic Regression========#
def LogisticClass(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2')
    clf.fit(x_train, y_train)
    return clf


# ========KNN========#
def KnnClass(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    return clf


# ========Decision Tree ========#
def DccisionClass(x_train, y_train):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf


# ========Random Forest Classifier ========#
def random_forest_class(x_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=8)  # parameter: n_estimators is to set the number of weak classifier.
    clf.fit(x_train, y_train)
    return clf


# ========Evaluation========#
def Precision(clf):
    doc_class_predicted = clf.predict(x_test)
    print(np.mean(doc_class_predicted == y_test))  # predict label and true label
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
    answer = clf.predict_proba(x_test)[:, 1]
    report = answer > 0.5
    print(classification_report(y_test, report, target_names=['neg', 'pos']))
    print("--------------------")
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, doc_class_predicted))
    C = confusion_matrix(y_test, doc_class_predicted)
    print("-------Confusion Matrix--------")
    print(C)


# ========Evaluation1 ========#
def Precision1(doc_class_predicted, report):
    print(np.mean(np.array(doc_class_predicted) == y_test))  # 预测结果和真实标签
    precision, recall, thresholds = precision_recall_curve(y_test, np.array(doc_class_predicted))
    print(classification_report(y_test, report, target_names=['neg', 'pos']))
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, doc_class_predicted))
    C = confusion_matrix(y_test, doc_class_predicted)
    print("-------Confusion Matrix--------")
    print(C)

if __name__ == '__main__':
    '''
    pos_path="rt-polarity.pos"
    neg_path = "rt-polarity.neg"
    tokenization(pos_path, neg_path)
    '''
    data = []
    labels = []
    with open("train.txt", "r")as file:
        for line in file:
            line = line[0:1]
            labels.append(line)
    with open("train.txt", "r")as file:
        for line in file:
            line = line[1:]
            data.append(line)
    x = np.array(data)
    labels = np.array(labels)
    labels = [int(i) for i in labels]
    movie_target = labels
    # transfer to vector
    count_vec = TfidfVectorizer(binary=False)
    # random split data as 80% for training data, 20% for testing data.
    x_train, x_test, y_train, y_test = train_test_split(x, movie_target, test_size=0.2)
    x_train = count_vec.fit_transform(x_train)
    x_test = count_vec.transform(x_test)



    print('************** Random Baseline ************  ')
    guess(y_test)
    print('************** Logistic Regression ************  ')
    Precision(LogisticClass(x_train, y_train))
    print('************** Support Vector Machine ************  ')
    Precision(SvmClass(x_train, y_train))
    print('************** Naive Bayes ************  ')
    Precision(NbClass(x_train, y_train))
    print('************** K-nearest neighbors************  ')
    Precision(KnnClass(x_train, y_train))
    print('************** Decision Tree ************  ')
    Precision(DccisionClass(x_train, y_train))
    print('************** Random Forest Classifier ************  ')
    Precision(random_forest_class(x_train, y_train))


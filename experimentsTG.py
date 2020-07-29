import pandas as pd
from math import floor
from random import shuffle
import numpy as np
import time

from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier

from joblib import Parallel, delayed

#load dataset and return dataframes
def load_data():
    df = pd.read_csv('reviews.csv', '¨')
    df = df.drop(['other', 'Star rating', 'Product'], 'columns')
    df['Functional'].replace('x', '_Label_Func', inplace=True)
    df['Performance'].replace('x', '_Label_Perf', inplace=True)
    df['Compatibility'].replace('x', '_Label_Comp', inplace=True)
    df['Usability'].replace('x', '_Label_Usab', inplace=True)
    df.fillna('_Label_Zero', inplace=True)

    columns = list(df.columns.values)
    columns.remove('Text')

    functional_df = df.drop(['Compatibility', 'Performance', 'Usability'], 'columns')
    performance_df = df.drop(['Compatibility', 'Functional', 'Usability'], 'columns')
    compatibility_df = df.drop(['Functional', 'Performance', 'Usability'], 'columns')
    usability_df = df.drop(['Compatibility', 'Performance', 'Functional'], 'columns')

    functional_df.rename(columns={'Functional': 'Label',
                    'Performance': 'Label',
                    'Compatibility': 'Label',
                    'Usability': 'Label'},
           inplace=True)
    compatibility_df.rename(columns={'Functional': 'Label',
                    'Performance': 'Label',
                    'Compatibility': 'Label',
                    'Usability': 'Label'},
           inplace=True)
    usability_df.rename(columns={'Functional': 'Label',
                    'Performance': 'Label',
                    'Compatibility': 'Label',
                    'Usability': 'Label'},
           inplace=True)
    performance_df.rename(columns={'Functional': 'Label',
                    'Performance': 'Label',
                    'Compatibility': 'Label',
                    'Usability': 'Label'},
           inplace=True)

    dfs = []
    dfs.append(functional_df)
    dfs.append(performance_df)
    dfs.append(compatibility_df)
    dfs.append(usability_df)
    return dfs, columns

def getIndexesUS(df): #10fold Cross Validation with under sampling
    label_0s = df.index[df['Label'] == '_Label_Zero'].tolist()  # contains indexes which column label matches label_zero
    label_relevants = df.index[df['Label'] != '_Label_Zero'].tolist()
    #shuffle(label_0s)
    #shuffle(label_relevants)
    original_ratio = len(label_relevants)/len(label_0s)
    list_of_training_indexes_lists = []
    list_of_testing_indexes_lists = []
    label_relevants_fold_size = floor(len(label_relevants) / 10)
    label_0_fold_size = floor(label_relevants_fold_size / original_ratio)
    for i in range(10):
        # number of irrelevant features to retrieve from label_0 is 9/10*fold_size because of 10 fold cross validation
        label_0s_downsample_train = label_0s[:i * label_0_fold_size]
        label_0s_downsample_train.extend(label_0s[(i + 1) * label_0_fold_size:10 * label_0_fold_size])
        label_relevants_train = label_relevants[:i * label_relevants_fold_size]
        label_relevants_train.extend(label_relevants[(i + 1) * label_relevants_fold_size:10 * label_relevants_fold_size])
        training_indexes = label_0s_downsample_train
        training_indexes.extend(label_relevants_train)
        #shuffle(training_indexes)
        list_of_training_indexes_lists.append(training_indexes)

        #shuffle(label_0s)
        label_0s_test = label_0s[i * label_0_fold_size:(i+1)*label_0_fold_size]
        label_relevants_test = label_relevants[i * label_relevants_fold_size:(i + 1) * label_relevants_fold_size]
        testing_indexes = label_relevants_test
        testing_indexes.extend(label_0s_test)
        #shuffle(testing_indexes)
        list_of_testing_indexes_lists.append(testing_indexes)
    return list_of_training_indexes_lists, list_of_testing_indexes_lists

def getIndexesCV(df): #Normal 10fold Cross-Validation
    indexes = [item for item in range(len(df))]
    #shuffle(indexes) #gives error if not commented, dunno why
    fold_size = floor(len(indexes) / 10) #1500 / 10 = 150
    list_of_training_indexes_lists = []
    list_of_testing_indexes_lists = []

    for i in range(10):
        labels_train_fold = indexes[:i * fold_size]
        labels_train_fold.extend(indexes[(i + 1) * fold_size:])
        list_of_training_indexes_lists.append(labels_train_fold)

        labels_test_fold = indexes[i * fold_size:(i + 1) * fold_size]
        list_of_testing_indexes_lists.append(labels_test_fold)
    return list_of_training_indexes_lists, list_of_testing_indexes_lists

def confusionMatrix(predictions, real_labels):
    TrueNegatives = 0
    FalseNegatives = 0
    TruePositives = 0
    FalsePositives = 0

    predictions = predictions.tolist()
    if(not isinstance(real_labels, list)):
        real_labels = real_labels.tolist()

    for i in range(len(predictions)):
        if(predictions[i] == "_Label_Zero"):
            if(predictions[i] == real_labels[i]):
                TrueNegatives +=1
            else:
                FalseNegatives+=1
        else:
            if(predictions[i] == real_labels[i]):
                TruePositives +=1
            else:
                FalsePositives +=1
    return TruePositives, TrueNegatives, FalsePositives, FalseNegatives

def evaluate(TruePositives, TrueNegatives, FalsePositives, FalseNegatives):
    if(TruePositives + FalsePositives == 0):
        precision = -1
    else:
        precision = (TruePositives)/(TruePositives + FalsePositives)
    if (TruePositives + FalseNegatives == 0):
        recall = -1
    else:
        recall = TruePositives/(TruePositives + FalseNegatives)
    f1measure = 2*precision*recall/(precision + recall)

    f2measure = 5*TruePositives/(5*TruePositives + 4*FalseNegatives + FalsePositives)

    return precision, recall, f1measure, f2measure

def printConfusionMatrix(TruePositives, TrueNegatives, FalsePositives, FalseNegatives):
    print("Confusion Matrix:")
    print("\t"+str(TruePositives) + "|" + str(FalseNegatives))
    print("\t"+str(FalsePositives)+ "|" + str(TrueNegatives)+"\n")

def printEvalCost(TruePositives, TrueNegatives, FalsePositives, FalseNegatives, ratio):
    cost = 0*TruePositives + 0*TrueNegatives + 1*FalsePositives + FalseNegatives/ratio
    print("Cost: " + str(cost) + "\n")

# reset counters
TP_NB, TN_NB, FP_NB, FN_NB = 0, 0, 0, 0
TP_NBtf, TN_NBtf, FP_NBtf, FN_NBtf = 0, 0, 0, 0
TP_TREE, TN_TREE, FP_TREE, FN_TREE = 0, 0, 0, 0
TP_TREEtf, TN_TREEtf, FP_TREEtf, FN_TREEtf = 0, 0, 0, 0
TP_SVM, TN_SVM, FP_SVM, FN_SVM = 0, 0, 0, 0
TP_SVMtf, TN_SVMtf, FP_SVMtf, FN_SVMtf = 0, 0, 0, 0
TP_LR, TN_LR, FP_LR, FN_LR = 0, 0, 0, 0
TP_LRtf, TN_LRtf, FP_LRtf, FN_LRtf = 0, 0, 0, 0
TP_RF, TN_RF, FP_RF, FN_RF = 0, 0, 0, 0
TP_RFtf, TN_RFtf, FP_RFtf, FN_RFtf = 0, 0, 0, 0
TP_Grad, TN_Grad, FP_Grad, FN_Grad = 0, 0, 0, 0
TP_Gradtf, TN_Gradtf, FP_Gradtf, FN_Gradtf = 0, 0, 0, 0
#unigrams = []
pred_list = []
text_list = []

def thread_code(j, getIndexes, df, fold_size_0, fold_size_relevant, i, costLearningWeight):
    global TP_NB, TN_NB, FP_NB, FN_NB, TP_NBtf, TN_NBtf, FP_NBtf, FN_NBtf,TP_TREE, TN_TREE, FP_TREE, FN_TREE,TP_TREEtf, TN_TREEtf, FP_TREEtf, FN_TREEtf,TP_SVM, TN_SVM, FP_SVM, FN_SVM,TP_SVMtf, TN_SVMtf, FP_SVMtf, FN_SVMtf,TP_LR, TN_LR, FP_LR, FN_LR,TP_LRtf, TN_LRtf, FP_LRtf, FN_LRtf,TP_RF, TN_RF, FP_RF, FN_RF,TP_RFtf, TN_RFtf, FP_RFtf, FN_RFtf, TP_Grad, TN_Grad, FP_Grad, FN_Grad, TP_Gradtf, TN_Gradtf, FP_Gradtf, FN_Gradtf, pred_list, text_list #, unigrams

    label_0s = df.index[df['Label'] == '_Label_Zero'].tolist()  # contains indexes which column label matches label_zero
    label_relevants = df.index[df['Label'] != '_Label_Zero'].tolist()
    if (getIndexes == False):  # IF SMOTE:

        index_train_zero = label_0s[:j * fold_size_0]
        index_train_zero.extend(label_0s[(j + 1) * fold_size_0:])
        index_train_relevant = label_relevants[:j * fold_size_relevant]
        index_train_relevant.extend(label_relevants[(j + 1) * fold_size_relevant:])
        index_train = index_train_relevant
        index_train.extend(index_train_zero)

        test_zero = label_0s[j * fold_size_0:(j + 1) * fold_size_0]
        test_relevant = label_relevants[j * fold_size_relevant:(j + 1) * fold_size_relevant]
        index_test = test_zero
        index_test.extend(test_relevant)

        corpus = df['Text']
        corpus_train = [corpus[k] for k in index_train]
        corpus_test = [corpus[k] for k in index_test]
        labels = df['Label']
        labels_train = [labels[k] for k in index_train]
        labels_test = [labels[k] for k in index_test]

        # Bag of Words
        vectorizer = CountVectorizer(stop_words='english')
        bow_train = vectorizer.fit_transform(corpus_train)
        bow_train = bow_train.toarray()
        bow_test = vectorizer.transform(corpus_test)
        # Term Frequency - Inverse Document Frequency
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf_train = transformer.fit_transform(bow_train)
        tfidf_train.toarray()
        tfidf_test = transformer.transform(bow_test)

        # select reviews
        smote = SMOTE(k_neighbors=3)
        bow_train, labels_train_bow = smote.fit_resample(bow_train, labels_train)
        tfidf_train, labels_train_tf = smote.fit_resample(tfidf_train, labels_train)
    else:
        trainll, testll = getIndexes(df)  # contains indexes for training and testing
        # select reviews
        corpus_train = df['Text'].loc[trainll[j]]
        corpus_test = df['Text'].loc[testll[j]]
        # select labels
        labels_train_bow = df['Label'].iloc[trainll[j]]
        labels_train_tf = labels_train_bow
        labels_test = df['Label'].iloc[testll[j]]

        # Bag of Words
        vectorizer = CountVectorizer(stop_words='english')
        bow_train = vectorizer.fit_transform(corpus_train)
        bow_train = bow_train.toarray()
        bow_test = vectorizer.transform(corpus_test)

        # Term Frequency - Inverse Document Frequency
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf_train = transformer.fit_transform(bow_train)
        tfidf_train.toarray()
        tfidf_test = transformer.transform(bow_test)

        '''
        # chi2 to select the best correlated terms
        features_chi2 = chi2(bow_train, labels_train_bow)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(vectorizer.get_feature_names())[indices]
        unigramst = [v for v in feature_names if len(v.split(' ')) == 1]
        unigrams.extend(unigramst[-5:])'''

    if (costLearningWeight != False):
        if(str(costLearningWeight) == "balanced"):
            weights = 'balanced'
        else:
            weights = {'_Label_Zero': 1, df.iloc[label_relevants[0]]['Label']: costLearningWeight}
        # train, predict and evaluate with Decision Tree - BOW
        TREE = tree.DecisionTreeClassifier(class_weight=weights)
        TREE.fit(bow_train, labels_train_bow)
        predictions = TREE.predict(bow_test)

        TP_TREEt, TN_TREEt, FP_TREEt, FN_TREEt = confusionMatrix(predictions, labels_test)
        TP_TREE += TP_TREEt
        TN_TREE += TN_TREEt
        FP_TREE += FP_TREEt
        FN_TREE += FN_TREEt

        # train, predict and evaluate with Decision Tree - BOW
        TREE = tree.DecisionTreeClassifier(class_weight=weights)
        TREE.fit(tfidf_train, labels_train_tf)
        predictions = TREE.predict(tfidf_test)
        TP_TREEtft, TN_TREEtft, FP_TREEtft, FN_TREEtft = confusionMatrix(predictions, labels_test)
        TP_TREEtf += TP_TREEtft
        TN_TREEtf += TN_TREEtft
        FP_TREEtf += FP_TREEtft
        FN_TREEtf += FN_TREEtft

        # train, predict and evaluate with Decision Tree - TFIDF
        SVM = LinearSVC(class_weight=weights)
        SVM.fit(bow_train, labels_train_bow)
        predictions = SVM.predict(bow_test)
        TP_SVMt, TN_SVMt, FP_SVMt, FN_SVMt = confusionMatrix(predictions, labels_test)
        TP_SVM += TP_SVMt
        TN_SVM += TN_SVMt
        FP_SVM += FP_SVMt
        FN_SVM += FN_SVMt

        # train, predict and evaluate with SVM - TFIDF
        SVM = LinearSVC(class_weight=weights)
        SVM.fit(tfidf_train, labels_train_tf)
        predictions = SVM.predict(tfidf_test)
        TP_SVMtft, TN_SVMtft, FP_SVMtft, FN_SVMtft = confusionMatrix(predictions, labels_test)
        TP_SVMtf += TP_SVMtft
        TN_SVMtf += TN_SVMtft
        FP_SVMtf += FP_SVMtft
        FN_SVMtf += FN_SVMtft

        # train, predict and evaluate with Logistic Regression - BOW
        LR = LogisticRegression(random_state=0, class_weight=weights)
        LR.fit(bow_train, labels_train_bow)
        predictions = LR.predict(bow_test)
        TP_LRt, TN_LRt, FP_LRt, FN_LRt = confusionMatrix(predictions, labels_test)
        TP_LR += TP_LRt
        TN_LR += TN_LRt
        FP_LR += FP_LRt
        FN_LR += FN_LRt

        # train, predict and evaluate with Logistic Regression - TFIDF
        LR = LogisticRegression(random_state=0, class_weight=weights)
        LR.fit(tfidf_train, labels_train_tf)
        predictions = LR.predict(tfidf_test)
        text_list.extend(corpus_test)
        pred_list.extend(predictions)
        TP_LRtft, TN_LRtft, FP_LRtft, FN_LRtft = confusionMatrix(predictions, labels_test)
        TP_LRtf += TP_LRtft
        TN_LRtf += TN_LRtft
        FP_LRtf += FP_LRtft
        FN_LRtf += FN_LRtft
    else:
        # train, predict and evaluate with Multinomial Naive Bayes - BOW
        naive_bayes = MultinomialNB()
        naive_bayes.fit(bow_train, labels_train_bow)
        predictions = naive_bayes.predict(bow_test)
        text_list.extend(corpus_test)
        pred_list.extend(predictions)
        TP_NBt, TN_NBt, FP_NBt, FN_NBt = confusionMatrix(predictions, labels_test)
        TP_NB += TP_NBt
        TN_NB += TN_NBt
        FP_NB += FP_NBt
        FN_NB += FN_NBt

        # train, predict and evaluate with Multinomial Naive Bayes - TFIDF
        naive_bayes = MultinomialNB()
        naive_bayes.fit(tfidf_train, labels_train_tf)
        predictions = naive_bayes.predict(tfidf_test)
        TP_NBtft, TN_NBtft, FP_NBtft, FN_NBtft = confusionMatrix(predictions, labels_test)
        TP_NBtf += TP_NBtft
        TN_NBtf += TN_NBtft
        FP_NBtf += FP_NBtft
        FN_NBtf += FN_NBtft

        # train, predict and evaluate with SVM - BOW
        SVM = LinearSVC()
        SVM.fit(bow_train, labels_train_bow)
        predictions = SVM.predict(bow_test)
        TP_SVMt, TN_SVMt, FP_SVMt, FN_SVMt = confusionMatrix(predictions, labels_test)
        TP_SVM += TP_SVMt
        TN_SVM += TN_SVMt
        FP_SVM += FP_SVMt
        FN_SVM += FN_SVMt

        # train, predict and evaluate with SVM - TFIDF
        SVM = LinearSVC()
        SVM.fit(tfidf_train, labels_train_tf)
        predictions = SVM.predict(tfidf_test)
        TP_SVMtft, TN_SVMtft, FP_SVMtft, FN_SVMtft = confusionMatrix(predictions, labels_test)
        TP_SVMtf += TP_SVMtft
        TN_SVMtf += TN_SVMtft
        FP_SVMtf += FP_SVMtft
        FN_SVMtf += FN_SVMtft

        # train, predict and evaluate with Logistic Regression - BOW
        LR = LogisticRegression(random_state=0)
        LR.fit(bow_train, labels_train_bow)
        predictions = LR.predict(bow_test)
        TP_LRt, TN_LRt, FP_LRt, FN_LRt = confusionMatrix(predictions, labels_test)
        TP_LR += TP_LRt
        TN_LR += TN_LRt
        FP_LR += FP_LRt
        FN_LR += FN_LRt

        # train, predict and evaluate with Logistic Regression - TFIDF
        LR = LogisticRegression(random_state=0)
        LR.fit(tfidf_train, labels_train_tf)
        predictions = LR.predict(tfidf_test)
        TP_LRtft, TN_LRtft, FP_LRtft, FN_LRtft = confusionMatrix(predictions, labels_test)
        TP_LRtf += TP_LRtft
        TN_LRtf += TN_LRtft
        FP_LRtf += FP_LRtft
        FN_LRtf += FN_LRtft

        # train, predict and evaluate with Random Forest - BOW
        RF = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
        RF.fit(bow_train, labels_train_bow)
        predictions = RF.predict(bow_test)
        TP_RFt, TN_RFt, FP_RFt, FN_RFt = confusionMatrix(predictions, labels_test)
        TP_RF += TP_RFt
        TN_RF += TN_RFt
        FP_RF += FP_RFt
        FN_RF += FN_RFt

        # train, predict and evaluate with Random Forest - TFIDF
        RF = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
        RF.fit(tfidf_train, labels_train_tf)
        predictions = RF.predict(tfidf_test)
        TP_RFtft, TN_RFtft, FP_RFtft, FN_RFtft = confusionMatrix(predictions, labels_test)
        TP_RFtf += TP_RFtft
        TN_RFtf += TN_RFtft
        FP_RFtf += FP_RFtft
        FN_RFtf += FN_RFtft

        Grad = GradientBoostingClassifier()
        Grad.fit(bow_train, labels_train_tf)
        predictions = Grad.predict(bow_test)
        TP_Gradt, TN_Gradt, FP_Gradt, FN_Gradt = confusionMatrix(predictions, labels_test)
        TP_Grad += TP_Gradt
        TN_Grad += TN_Gradt
        FP_Grad += FP_Gradt
        FN_Grad += FN_Gradt

        Grad = GradientBoostingClassifier()
        Grad.fit(tfidf_train, labels_train_tf)
        predictions = Grad.predict(tfidf_test)
        TP_Gradtft, TN_Gradtft, FP_Gradtft, FN_Gradtft = confusionMatrix(predictions, labels_test)
        TP_Gradtf += TP_Gradtft
        TN_Gradtf += TN_Gradtft
        FP_Gradtf += FP_Gradtft
        FN_Gradtf += FN_Gradtft

def print_evaluation(dfs, classes, getIndexes, title, costLearningWeight):
    global TP_NB, TN_NB, FP_NB, FN_NB, TP_NBtf, TN_NBtf, FP_NBtf, FN_NBtf,TP_TREE, TN_TREE, FP_TREE, FN_TREE,TP_TREEtf, TN_TREEtf, FP_TREEtf, FN_TREEtf,TP_SVM, TN_SVM, FP_SVM, FN_SVM,TP_SVMtf, TN_SVMtf, FP_SVMtf, FN_SVMtf,TP_LR, TN_LR, FP_LR, FN_LR,TP_LRtf, TN_LRtf, FP_LRtf, FN_LRtf,TP_RF, TN_RF, FP_RF, FN_RF,TP_RFtf, TN_RFtf, FP_RFtf, FN_RFtf, TP_Grad, TN_Grad, FP_Grad, FN_Grad, TP_Gradtf, TN_Gradtf, FP_Gradtf, FN_Gradtf, pred_list, text_list #, unigrams
    print("============================================================================")
    print(title)
    for i in range(len(dfs)): #each dataframe corresponds to a different class
        df = dfs[i]
        label = classes[i]

        label_0s = df.index[df['Label'] == '_Label_Zero'].tolist()
        label_relevants = df.index[df['Label'] != '_Label_Zero'].tolist()
        ratio = len(label_relevants)/len(label_0s)
        fold_size_0 = floor(len(label_0s) / 10)
        fold_size_relevant = floor(len(label_relevants) / 10)

        #reset counters
        TP_NB, TN_NB, FP_NB, FN_NB = 0, 0, 0, 0
        TP_NBtf, TN_NBtf, FP_NBtf, FN_NBtf = 0, 0, 0, 0
        TP_TREE, TN_TREE, FP_TREE, FN_TREE = 0, 0, 0, 0
        TP_TREEtf, TN_TREEtf, FP_TREEtf, FN_TREEtf = 0, 0, 0, 0
        TP_SVM, TN_SVM, FP_SVM, FN_SVM = 0, 0, 0, 0
        TP_SVMtf, TN_SVMtf, FP_SVMtf, FN_SVMtf = 0, 0, 0, 0
        TP_LR, TN_LR, FP_LR, FN_LR = 0, 0, 0, 0
        TP_LRtf, TN_LRtf, FP_LRtf, FN_LRtf = 0, 0, 0, 0
        TP_RF, TN_RF, FP_RF, FN_RF = 0, 0, 0, 0
        TP_RFtf, TN_RFtf, FP_RFtf, FN_RFtf = 0, 0, 0, 0
        TP_Grad, TN_Grad, FP_Grad, FN_Grad = 0, 0, 0, 0
        TP_Gradtf, TN_Gradtf, FP_Gradtf, FN_Gradtf = 0, 0, 0, 0
        pred_list = [] #contains the predicted labels
        text_list = [] #contains the predicted texts

        print("----------------------------------------------------------------------------")
        print("label: " + str(label))

        Parallel(n_jobs=-1, require='sharedmem')(delayed(thread_code)(j, getIndexes, df, fold_size_0, fold_size_relevant, i, costLearningWeight) for j in range(10))

        if (costLearningWeight != False):
            #preparation for chi2
            newdf = pd.DataFrame(list(zip(text_list, pred_list)),
                         columns=['Text', 'Label'])
            corpus_pred = newdf['Text']
            labels_pred = newdf['Label']
            vectorizer = CountVectorizer(stop_words='english')
            bow_pred = vectorizer.fit_transform(corpus_pred)
            bow_pred = bow_pred.toarray()

            # chi2 to select the best correlated terms for the predicted labels
            features_chi2 = chi2(bow_pred, labels_pred)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(vectorizer.get_feature_names())[indices]
            unigramst = [v for v in feature_names if len(v.split(' ')) == 1]
            unigrams = list(dict.fromkeys(unigramst[-5:]))
            print("  . Most correlated unigrams:\n." +str(unigrams))

            print(pred_list)
            #pred_file = pd.DataFrame(text_list, pred_list)
            newdf.to_csv(str(label) + '_predfile.csv', index=False, sep=';')

            precisionSVM, recallSVM, fmeasureSVM, f2measureSVM = evaluate(TP_SVM, TN_SVM, FP_SVM, FN_SVM)
            print("SVM - BOW:\n\tPrecision = " + str(precisionSVM) + "\n\tRecall = " + str(
                recallSVM) + "\n\tF-Measure = " + str(fmeasureSVM)+ "\n\tF2-Measure = " + str(f2measureSVM))
            printConfusionMatrix(TP_SVM, TN_SVM, FP_SVM, FN_SVM)
            printEvalCost(TP_SVM, TN_SVM, FP_SVM, FN_SVM, ratio)

            precisionSVMtf, recallSVMtf, fmeasureSVMtf, f2measureSVMtf = evaluate(TP_SVMtf, TN_SVMtf, FP_SVMtf, FN_SVMtf)
            print("SVM - TF-IDF:\n\tPrecision = " + str(precisionSVMtf) + "\n\tRecall = " + str(
                recallSVMtf) + "\n\tF-Measure = " + str(fmeasureSVMtf)+ "\n\tF2-Measure = " + str(f2measureSVMtf))
            printConfusionMatrix(TP_SVMtf, TN_SVMtf, FP_SVMtf, FN_SVMtf)
            printEvalCost(TP_SVMtf, TN_SVMtf, FP_SVMtf, FN_SVMtf, ratio)

            precisionTREE, recallTREE, fmeasureTREE, f2measureTREE = evaluate(TP_TREE, TN_TREE, FP_TREE, FN_TREE)
            print("TREE - BOW:\n\tPrecision = " + str(precisionTREE) + "\n\tRecall = " + str(
                recallTREE) + "\n\tF-Measure = " + str(fmeasureTREE)+ "\n\tF2-Measure = " + str(f2measureTREE))
            printConfusionMatrix(TP_TREE, TN_TREE, FP_TREE, FN_TREE)
            printEvalCost(TP_TREE, TN_TREE, FP_TREE, FN_TREE, ratio)

            precisionTREEtf, recallTREEtf, fmeasureTREEtf, f2measureTREEtf = evaluate(TP_TREEtf, TN_TREEtf, FP_TREEtf, FN_TREEtf)
            print("TREE - TF-IDF:\n\tPrecision = " + str(precisionTREEtf) + "\n\tRecall = " + str(
                recallTREEtf) + "\n\tF-Measure = " + str(fmeasureTREEtf)+ "\n\tF2-Measure = " + str(f2measureTREEtf))
            printConfusionMatrix(TP_TREEtf, TN_TREEtf, FP_TREEtf, FN_TREEtf)
            printEvalCost(TP_TREEtf, TN_TREEtf, FP_TREEtf, FN_TREEtf, ratio)

            precisionLR, recallLR, fmeasureLR, f2measureLR = evaluate(TP_LR, TN_LR, FP_LR, FN_LR)
            print("LR - BOW:\n\tPrecision = " + str(precisionLR) + "\n\tRecall = " + str(
                recallLR) + "\n\tF-Measure = " + str(fmeasureLR)+ "\n\tF2-Measure = " + str(f2measureLR))
            printConfusionMatrix(TP_LR, TN_LR, FP_LR, FN_LR)
            printEvalCost(TP_LR, TN_LR, FP_LR, FN_LR, ratio)

            precisionLRtf, recallLRtf, fmeasureLRtf, f2measureLRtf = evaluate(TP_LRtf, TN_LRtf, FP_LRtf, FN_LRtf)
            print("LR - TF-IDF:\n\tPrecision = " + str(precisionLRtf) + "\n\tRecall = " + str(
                recallLRtf) + "\n\tF-Measure = " + str(fmeasureLRtf)+ "\n\tF2-Measure = " + str(f2measureLRtf))
            printConfusionMatrix(TP_LRtf, TN_LRtf, FP_LRtf, FN_LRtf)
            printEvalCost(TP_LRtf, TN_LRtf, FP_LRtf, FN_LRtf, ratio)
        else:
            #unigrams = list(dict.fromkeys(unigrams))
            #print("  . Most correlated unigrams:\n." +str(unigrams))
            newdf = pd.DataFrame(list(zip(text_list, pred_list)),
                                 columns=['Text', 'Label'])
            newdf.to_csv(str(label) + '_predfile.csv', index=False, sep=';')

            precisionNB, recallNB, fmeasureNB, f2measureNB = evaluate(TP_NB, TN_NB, FP_NB, FN_NB)
            print("Naive Bayes - BOW:\n\tPrecision = "+ str(precisionNB) + "\n\tRecall = "+str(recallNB) + "\n\tF-Measure = "+str(fmeasureNB)+ "\n\tF2-Measure = " + str(f2measureNB))
            printConfusionMatrix(TP_NB, TN_NB, FP_NB, FN_NB)

            precisionNBtf, recallNBtf, fmeasureNBtf, f2measureNBtf= evaluate(TP_NBtf, TN_NBtf, FP_NBtf, FN_NBtf)
            print("Naive Bayes - TF-IDF:\n\tPrecision = "+ str(precisionNBtf) + "\n\tRecall = "+str(recallNBtf) + "\n\tF-Measure = "+str(fmeasureNBtf)+ "\n\tF2-Measure = " + str(f2measureNBtf))
            printConfusionMatrix(TP_NBtf, TN_NBtf, FP_NBtf, FN_NBtf)

            precisionSVM, recallSVM, fmeasureSVM, f2measureSVM = evaluate(TP_SVM, TN_SVM, FP_SVM, FN_SVM)
            print("SVM - BOW:\n\tPrecision = "+ str(precisionSVM) + "\n\tRecall = "+str(recallSVM) + "\n\tF-Measure = "+str(fmeasureSVM)+ "\n\tF2-Measure = " + str(f2measureSVM))
            printConfusionMatrix(TP_SVM, TN_SVM, FP_SVM, FN_SVM)

            precisionSVMtf, recallSVMtf, fmeasureSVMtf, f2measureSVMtf = evaluate(TP_SVMtf, TN_SVMtf, FP_SVMtf, FN_SVMtf)
            print("SVM - TF-IDF:\n\tPrecision = "+ str(precisionSVMtf) + "\n\tRecall = "+str(recallSVMtf) + "\n\tF-Measure = "+str(fmeasureSVMtf)+ "\n\tF2-Measure = " + str(f2measureSVMtf))
            printConfusionMatrix(TP_SVMtf, TN_SVMtf, FP_SVMtf, FN_SVMtf)

            precisionLR, recallLR, fmeasureLR, f2measureLR = evaluate(TP_LR, TN_LR, FP_LR, FN_LR)
            print("LR - BOW:\n\tPrecision = " + str(precisionLR) + "\n\tRecall = " + str(recallLR) + "\n\tF-Measure = " + str(fmeasureLR)+ "\n\tF2-Measure = " + str(f2measureLR))
            printConfusionMatrix(TP_LR, TN_LR, FP_LR, FN_LR)

            precisionLRtf, recallLRtf, fmeasureLRtf, f2measureLRtf = evaluate(TP_LRtf, TN_LRtf, FP_LRtf, FN_LRtf)
            print("LR - TF-IDF:\n\tPrecision = " + str(precisionLRtf) + "\n\tRecall = " + str(recallLRtf) + "\n\tF-Measure = " + str(fmeasureLRtf)+ "\n\tF2-Measure = " + str(f2measureLRtf))
            printConfusionMatrix(TP_LRtf, TN_LRtf, FP_LRtf, FN_LRtf)

            precisionRF, recallRF, fmeasureRF, f2measureRF = evaluate(TP_RF, TN_RF, FP_RF, FN_RF)
            print("RF - BOW:\n\tPrecision = " + str(precisionRF) + "\n\tRecall = " + str(recallRF) + "\n\tF-Measure = " + str(fmeasureRF)+ "\n\tF2-Measure = " + str(f2measureRF))
            printConfusionMatrix(TP_RF, TN_RF, FP_RF, FN_RF)

            precisionRFtf, recallRFtf, fmeasureRFtf, f2measureRFtf = evaluate(TP_RFtf, TN_RFtf, FP_RFtf, FN_RFtf)
            print("RF - TF-IDF:\n\tPrecision = " + str(precisionRFtf) + "\n\tRecall = " + str(recallRFtf) + "\n\tF-Measure = " + str(fmeasureRFtf)+ "\n\tF2-Measure = " + str(f2measureRFtf))
            printConfusionMatrix(TP_RFtf, TN_RFtf, FP_RFtf, FN_RFtf)

            precisionGrad, recallGrad, fmeasureGrad, f2measureGrad = evaluate(TP_Grad, TN_Grad, FP_Grad, FN_Grad)
            print("Grad - TF-IDF:\n\tPrecision = " + str(precisionGrad) + "\n\tRecall = " + str(
                recallGrad) + "\n\tF-Measure = " + str(fmeasureGrad) + "\n\tF2-Measure = " + str(f2measureGrad))
            printConfusionMatrix(TP_Grad, TN_Grad, FP_Grad, FN_Grad)

            precisionGradtf, recallGradtf, fmeasureGradtf, f2measureGradtf = evaluate(TP_Gradtf, TN_Gradtf, FP_Gradtf, FN_Gradtf)
            print("Grad - TF-IDF:\n\tPrecision = " + str(precisionGradtf) + "\n\tRecall = " + str(
                recallGradtf) + "\n\tF-Measure = " + str(fmeasureGradtf) + "\n\tF2-Measure = " + str(f2measureGradtf))
            printConfusionMatrix(TP_Gradtf, TN_Gradtf, FP_Gradtf, FN_Gradtf)

#main
start_time = time.time()
dfs, indexes = load_data()

print_evaluation(dfs, indexes, getIndexesCV, "Normal 10fold - Cross Validation", False) #last parameter stands for cost sensitive learning mode = off
print_evaluation(dfs, indexes, getIndexesUS, "10fold - Cross Validation with UnderSampling", False)
print_evaluation(dfs, indexes, False, "10fold - Cross Validation with SMOTE", False)    #"Get indexes = False" stands for "smote_mode = on"
print_evaluation(dfs, indexes, getIndexesCV, "Cost Sensitive Learning \'balanced\' > 10fold - Cross Validation", "balanced")
print_evaluation(dfs, indexes, getIndexesCV, "Cost Sensitive Learning 1:2 > 10fold - Cross Validation", 2)
print_evaluation(dfs, indexes, getIndexesCV, "Cost Sensitive Learning 1:5 > 10fold - Cross Validation", 5)
print_evaluation(dfs, indexes, getIndexesCV, "Cost Sensitive Learning 1:10 > 10fold - Cross Validation", 10)


print("Execution Time: " + str(floor((time.time() - start_time)/60)) + "min " + str((time.time() - start_time)%60) + "sec")

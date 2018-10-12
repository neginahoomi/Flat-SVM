from sklearn.preprocessing import MultiLabelBinarizer
import csv
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import KFold
import time
from sklearn.naive_bayes import BernoulliNB   
     
start_time = time.time()    
        
if __name__ == "__main__":
    
    titles = []
    firstlevel = []
    secondlevel = []
    thirdlevel = []
    fourthlevel = []
    labels_0 = []

    with open ('train_set.csv','rb') as infile,open('level_4.csv','wb') as out4,open('level_3.csv','wb') as out3,open('level_2.csv','wb') as out2,open('level_1.csv','wb') as out1:
        #reading data
        reader = csv.reader(infile)
        writer4 = csv.writer(out4)
        writer3 = csv.writer(out3)
        writer2 = csv.writer(out2)
        writer1 = csv.writer(out1)

        mydict = {rows[0]:rows[1:56] for rows in reader}
        #preparing multi label data
        for keys,values in mydict.items():
            titles.append(keys)
            labels_0.append(values)            
        titles_array= np.asarray(titles)
        
        for element in labels_0:
            element = [int(x) for x in element if x !='']
            element = list(set(element))
            fourthlevel.append(element)   
                     
            #creating third level labels
            element = [int(np.floor(x/10)) for x in element]
            element = list(set(element))
            thirdlevel.append(element)
            
            element = [int(np.floor(x/100)) for x in element]
            element = list(set(element))
            secondlevel.append(element)
            
            element = [int(np.floor(x/10)) for x in element]
            element = list(set(element))
            firstlevel.append(element)
            
        #dealing with multi labels 
        mlb1 = MultiLabelBinarizer()  
        mlb2 = MultiLabelBinarizer() 
        mlb3 = MultiLabelBinarizer()
        mlb4 = MultiLabelBinarizer()   
        firstlevel_coded = mlb1.fit_transform(firstlevel) 
        secondlevel_coded = mlb2.fit_transform(secondlevel)               
        thirdlevel_coded = mlb3.fit_transform(thirdlevel)
        fourthlevel_coded = mlb4.fit_transform(fourthlevel)
        
        kf = KFold(n_splits=10, random_state=0.1) #since the data has class imbalance you should use stratified sampling
        count_vect = CountVectorizer()        
        #clf =OneVsRestClassifier(LinearSVC(random_state = 1, C=2))
        #clf = OneVsRestClassifier(LogisticRegression(random_state= 1, C=16))
        clf = OneVsRestClassifier(BernoulliNB(alpha=0))
        
        for train_index, test_index in kf.split(titles_array):
            x_train, x_test = titles_array[train_index], titles_array[test_index]
            y_train_1, y_test_1 = firstlevel_coded[train_index], firstlevel_coded[test_index]
            y_train_2, y_test_2 = secondlevel_coded[train_index], secondlevel_coded[test_index]
            y_train_3, y_test_3 = thirdlevel_coded[train_index], thirdlevel_coded[test_index]                                    
            y_train_4, y_test_4 = fourthlevel_coded[train_index], fourthlevel_coded[test_index]
            x_train_count =  count_vect.fit_transform(x_train)
            x_test_count = count_vect.transform(x_test)
            #---------------------------------------------------------------------------------------------------------                       
            predicted_4th = clf.fit(x_train_count,y_train_4).predict(x_test_count) #predicted flat fourth level coded
	    microf1 = metrics.f1_score(y_test_4,predicted_4th,average = 'micro')
            macrof1 = metrics.f1_score(y_test_4,predicted_4th,average = 'macro')
            
            microp = metrics.precision_score(y_test_4,predicted_4th,average = 'micro')
            macrop = metrics.precision_score(y_test_4,predicted_4th,average = 'macro')
            
            micror = metrics.recall_score(y_test_4,predicted_4th,average = 'micro')
            macror = metrics.recall_score(y_test_4,predicted_4th,average = 'macro')
            writer4.writerow([microf1,macrof1,microp,macrop,micror,macror])
            #-------------------------------------------------------------------------------------------------------
            predicted_3 = clf.fit(x_train_count,y_train_3).predict(x_test_count) #predicted flat fourth level coded
	    microf1 = metrics.f1_score(y_test_3,predicted_3,average = 'micro')
            macrof1 = metrics.f1_score(y_test_3,predicted_3,average = 'macro')
            
            microp = metrics.precision_score(y_test_3,predicted_3,average = 'micro')
            macrop = metrics.precision_score(y_test_3,predicted_3,average = 'macro')
            
            micror = metrics.recall_score(y_test_3,predicted_3,average = 'micro')
            macror = metrics.recall_score(y_test_3,predicted_3,average = 'macro')
            writer3.writerow([microf1,macrof1,microp,macrop,micror,macror])
            #-------------------------------------------------------------------------------------------------------
            predicted_2 = clf.fit(x_train_count,y_train_2).predict(x_test_count) #predicted flat fourth level coded
	    microf1 = metrics.f1_score(y_test_2,predicted_2,average = 'micro')
            macrof1 = metrics.f1_score(y_test_2,predicted_2,average = 'macro')
            
            microp = metrics.precision_score(y_test_2,predicted_2,average = 'micro')
            macrop = metrics.precision_score(y_test_2,predicted_2,average = 'macro')
            
            micror = metrics.recall_score(y_test_2,predicted_2,average = 'micro')
            macror = metrics.recall_score(y_test_2,predicted_2,average = 'macro')
            writer2.writerow([microf1,macrof1,microp,macrop,micror,macror])
            #-------------------------------------------------------------------------------------------------------
            predicted_1 = clf.fit(x_train_count,y_train_1).predict(x_test_count) #predicted flat fourth level coded
	    microf1 = metrics.f1_score(y_test_1,predicted_1,average = 'micro')
            macrof1 = metrics.f1_score(y_test_1,predicted_1,average = 'macro')
                        
            microp = metrics.precision_score(y_test_1,predicted_1,average = 'micro')
            macrop = metrics.precision_score(y_test_1,predicted_1,average = 'macro')
            
            micror = metrics.recall_score(y_test_1,predicted_1,average = 'micro')
            macror = metrics.recall_score(y_test_1,predicted_1,average = 'macro')
            writer1.writerow([microf1,macrof1,microp,macrop,micror,macror])
            
    print("--- %s seconds ---" % (time.time() - start_time))




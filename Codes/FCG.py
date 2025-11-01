"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
find intra communiyt in each neuron memberships and do the whole mapping and retest 
"""

#from curses.ascii import NULL
from sklearn import metrics
from scipy.special import rel_entr, kl_div
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from fcmeans import FCM
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import random
import math
import itertools
import newSom
# unsupervised continus and discrete som
class FCG():
    """
     Unsupervised SOM with continusous data,and discrete data combination
     som the orignal som
     som_continuous when continusous = all train data som = som_continuous
     soms_discrete is the soms for each discrete features

    """
    def __init__(self, 
                 data_train_all, 
                 data_train_continuous,
                 data_train_discrete,
                 data_test_all,
                 data_test_continuous,
                 data_test_discrete,
                 label_train_all,                       
                 label_test_all,
                 topology_ratio
                 ):
        """
        Parameters

        """

        self.data_train_all = data_train_all
        self.data_test_all = data_test_all

        self.data_train_continuous = data_train_continuous   
        self.data_test_continuous = data_test_continuous
        self.data_train_discrete_unnormalized = data_train_discrete 
        self.data_test_discrete_unnormalized = data_test_discrete   



        self.train_label_all = label_train_all
        self.train_label_all = self.train_label_all.astype(int)
        self.test_label_all = label_test_all
        self.test_label_all = self.test_label_all.astype(int)


    def purity_score(self,scorename, y_true, y_pred):
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        if(scorename == "test_score_baseline" ):
            self.test_score_baseline_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            print("test_score_baseline_p{}".format(self.test_score_baseline_p ))
        if(scorename == "test_score_FCG" ):
            self.test_score_FCG_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
            print("test_score_FCG_p {}".format(self.test_score_FCG_p ))
     

    def nmiScore(self,scorename, y_true, y_pred):
        if(scorename == "test_score_baseline" ):
            self.test_score_baseline_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_score_baseline_n {}".format(self.test_score_baseline_n ))  
        if(scorename == "test_score_FCG" ):          
            self.test_score_FCG_n = normalized_mutual_info_score(y_true,y_pred) 
            print("test_score_FCG_n {}".format(self.test_score_FCG_n ))  



    
    def ariScore(self,scorename, y_true, y_pred):
        if(scorename == "test_score_baseline" ):
            self.test_score_baseline_a = adjusted_rand_score(y_true,y_pred)
            print("test_score_baseline_a {}".format(self.test_score_baseline_a ))  
        if(scorename == "test_score_FCG" ):          
            self.test_score_FCG_a = adjusted_rand_score(y_true,y_pred) 
            print("test_score_FCG_a {}".format(self.test_score_FCG_a ))  

    

    # from data index to  [[2,35,34,3,23],[211,12,2,1]] get cluster index [[0,0,1,1] [0,0,1]]
    def get_mapped_class_in_clusters(self,clusters_indexes,real_class_label):
        mapped_class_in_clusters = []
        #print(f"clusters_indexes {clusters_indexes} ")
        #initialize mapped_clases_in_clusters
        for i in range(0, len(clusters_indexes)):
            mapped_class_in_clusters.append([])

        for j in range(0, len(clusters_indexes)):
            for item in clusters_indexes[j]:
                mapped_class_in_clusters[j].append(real_class_label[item])
        #print(f"mapped_class_in_clusters {mapped_class_in_clusters}")
        # mapped_clases_in_clusters = [[1,2,1,2,1,1],[2,2,2,2],[0,1,0]]
       # for x in mapped_class_in_clusters:
       #    # print(f"x : {x}")
       #    print(f" x {self.realpropationofclasslabelinclusters(x,mapped_class_in_clusters.index(x))} ")
        return mapped_class_in_clusters


    def getLabelMapping(self,predicted_class_label_in_each_cluster,Wtype  = 0):
        """
         predicted_class_label  = [[1,2,1,1],[3,3,3]]  the value in is the true value in class_label
         it means that predicted cluster 0 is 1 in class lable, cluster label 2 is 3 in class label
        """
        predicted_label_convert_to_class_label = []
       
        for item in predicted_class_label_in_each_cluster:
            #print(f"item{item}" )   
            if item != []:
                # the first item is for cluster0       
                # transfer to true class value based on indices in predict lables       
                
                predicted_label_convert_to_class_label.append(self.getMaxRepeatedElements(item))
            else:
                # -1 means there is no data in current neuron
                predicted_label_convert_to_class_label.append(-1)
        
        if Wtype == 0 :
            #print(f"predicted_class_label_in_each_cluster 0 {predicted_class_label_in_each_cluster} ")
            self.PLabel_to_Tlabel_Mapping_W_baseline = predicted_label_convert_to_class_label
            #*********** remove null neurons in original null

            #self.removeNullNeuronsInW(predicted_label_convert_to_class_label,self.som_continuous.weights0)

        if Wtype == 1 :
            #print(f"predicted_class_label_in_each_cluster {predicted_class_label_in_each_cluster} ")
            self.PLabel_to_Tlabel_Mapping_W_Continous = predicted_label_convert_to_class_label

        if Wtype == 2 :
            self.PLabel_to_Tlabel_Mapping_W_Discrete = predicted_label_convert_to_class_label

    def getMaxRepeatedElements(self, list):
        #Count number of occurrences of each value in array of non-negative ints.
        counts = np.bincount(list)
       # print(f"counts  {counts}  list len{len(list)}")
        #print(f"self.realpropationofclasslabelinclusters(list) {self.realpropationofclasslabelinclusters(list)}")
        #Returns the indices of the maximum values along an axis.
        return np.argmax(counts)

    def convertPredictedLabelValue(self,predicted_cluster_labels, PLable_TLabel_Mapping):
        # PLabel_CLabel_Mapping the mapping of cluster label to class label
        # PLable_TLabel_Mapping size is the som.m*som.n* stop_split_num
        #print(f"predicted_cluster_labels {predicted_cluster_labels} PLable_TLabel_Mapping {PLable_TLabel_Mapping} ")
        for i in range(0,len(predicted_cluster_labels)):
            predicted_cluster_value =  predicted_cluster_labels[i]
            predicted_cluster_labels[i] = PLable_TLabel_Mapping[predicted_cluster_value]      

  
        return predicted_cluster_labels

    def transferClusterLabelToClassLabel(self, mapping ,predicted_cluster_labels): 
        """
        map = self.PLabel_to_Tlabel_Mapping_W_baseline self.PLabel_to_Tlabel_Mapping_W_continous
        """     
        predicted_class_labels =  self.convertPredictedLabelValue(predicted_cluster_labels,mapping)

        return predicted_class_labels


    def getScore(self,scorename, y_true, y_pred):
       # print(len(y_true))
       # print(len(y_pred))
        self.purity_score(scorename,y_true,y_pred)
        self.nmiScore(scorename,y_true,y_pred)
        self.ariScore(scorename,y_true,y_pred)

    def getScores(self, score_tpye, y_true, y_pred):
      
        #print(f" y_pred {y_pred}")
        if score_tpye  == "Baseline":
            self.accuracy_score_baseline = accuracy_score(y_true,y_pred)
            print(f"accuracy_score_baseline {self.accuracy_score_baseline}")
            self.recall_score_baseline = recall_score(y_true,y_pred,average='macro')
            print(f"recall_score_baseline {self.recall_score_baseline}")
            self.precision_score_baseline = precision_score(y_true,y_pred,average='macro')
            print(f"precision_score_baseline {self.precision_score_baseline}")
            self.f1_score_baseline = f1_score(y_true,y_pred,average='macro')
            print(f"f1_score_baseline {self.f1_score_baseline}")

        elif score_tpye  == "FCG":
            
            #print(f" y_true {y_true}")
            #print(f" y_pred {y_pred}")
            self.accuracy_score_fcg = accuracy_score(y_true,y_pred)
            print(f"accuracy_score_fcg {self.accuracy_score_fcg}")
            self.recall_score_fcg = recall_score(y_true,y_pred,average='macro')
            print(f"recall_score_fcg {self.recall_score_fcg}")
            self.precision_score_fcg = precision_score(y_true,y_pred,average='macro')
            print(f"precision_score_fcg {self.precision_score_fcg}")
            self.f1_score_fcg = f1_score(y_true,y_pred,average='macro')
            print(f"f1_score_fcg {self.f1_score_fcg}")

  
    def getCommanIndexesRatioInClassGroup_fuzzy(self, feature_group_intersection, one_class_group): # membership funciton
        # difference with getCommanIndexesRatioInNeurons is the denomitor
        if len(one_class_group) !=0:
            return round(len(np.intersect1d(feature_group_intersection, one_class_group))/len(one_class_group),3) 
        else: return 0

    def getCommanIndexesRatioInClassGroup_probability(self, feature_group_intersection, one_class_group): # membership funciton
        # difference with getCommanIndexesRatioInNeurons is the denomitor
        if len(one_class_group) !=0:
            return round(len(np.intersect1d(feature_group_intersection, one_class_group))/len(feature_group_intersection),3) 
        else: return 0

    def getFeatureGroups(self, feature_column_data):  #cluster feature data based on unique value
        #feature_group = dictionary{value:[indexes], value2: [indexes]} feature_column_data =
        feature_group = {}
       # print(f" feature_column_data{ feature_column_data} ")
        for i in range(0,len(feature_column_data)):
            if len(feature_group)>=1:  
                if  feature_column_data[i] in feature_group.keys():
                    feature_group[feature_column_data[i]].append(i) 
                else:
                    feature_group[feature_column_data[i]] = [i]
            else:
                 #print(f" i  {i}, self.feature_column_data[i] {feature_column_data[i]}")
                 feature_group[feature_column_data[i]] = [i]
        #print(f" feature_group{ feature_group} ") 
        return feature_group
    


        
    def getAllfeatureGroups(self):
        all_feature_groups = {}
       # print(f"np.shape(self.data_train_discrete_unnormalized)[1]{ np.shape(self.data_train_discrete_unnormalized)}")
        for i in range(0,np.shape(self.data_train_discrete_unnormalized)[1]):
            # i is the comlumn number
            all_feature_groups[i] = self.getFeatureGroups(self.data_train_discrete_unnormalized[:,i])

        return all_feature_groups
            #print(f" i {i} self.all_feature_gropus {  self.all_feature_groups[i] } ") #{15.0: [0, 1, 5, 6, 19, 12],1:[43,45,47]}
       # print(f"  self.all_feature_groups {  self.all_feature_groups}")

                    


    

    def getOneSingleFeatureClassProbability_fuzzy(self, one_feature_dic, class_groups_dic):
        onesinglefeature_to_class_probablity = {}
       
        for key in one_feature_dic.keys():
            probability_list = []
            for key2 in  class_groups_dic:  
                #print("New neuron _predicted group !!!")       
                probability = self.getCommanIndexesRatioInClassGroup_fuzzy(one_feature_dic[key],class_groups_dic[key2]) 
                #print(f"i {i} neuron_predicted_groups[i] {len(neuron_predicted_groups[i])}")  
                probability_list.append(probability)
            #print(f"key {key} probability_list{probability_list}")
            probability_list = np.array(probability_list)
            onesinglefeature_to_class_probablity[key] = probability_list
          #  print(f"key {key} probability_list fuzzy) sum {np.sum(probability_list)} ")
        return onesinglefeature_to_class_probablity



    


    def getEachClassProbabilityOfEachFeatureValue_fuzzy(self,class_groups_dic):
        self.all_features_mapping_fuzzy ={}
        for i in range(0, len(self.all_feature_groups)):
            self.all_features_mapping_fuzzy[i] = self.getOneSingleFeatureClassProbability_fuzzy(self.all_feature_groups[i],class_groups_dic)

        #print(f"  self.all_features_mapping_fuzzy { self.all_features_mapping_fuzzy}")


    def getEachNeuronProbabilityOfEachFeatureValue_fuzzy(self,neuron_predicted_groups):
        self.all_features_nuron_mapping_fuzzy ={}
        for i in range(0, len(self.all_feature_groups)):
           # print(f"  feature  {i}!!!!!!!!!!!")
            self.all_features_nuron_mapping_fuzzy[i] = self.getOneSingleFeatureNeuronProbability_fuzzy(self.all_feature_groups[i],neuron_predicted_groups)
    def getCommanIndexesRatioInNeurons_fuzzy(self, feature_group, one_neuron_predict_group):
        # difference with getCommanIndexesRatioInNeurons is the denomitor
        if len(one_neuron_predict_group) !=0:
            return round(len(np.intersect1d(feature_group, one_neuron_predict_group))/len(one_neuron_predict_group),3)
        else: return 0
    def getOneSingleFeatureNeuronProbability_fuzzy(self, one_feature_dic, neuron_predicted_groups):
        onesinglefeatureneuronprobablity = {}
        #print(f"neuron_predicted_groups {neuron_predicted_groups} ")
        for key in one_feature_dic.keys():
            probability_list = []
            for i in range(0,len(neuron_predicted_groups)) :  
                #print("New neuron _predicted group !!!")       
                probability = self.getCommanIndexesRatioInNeurons_fuzzy(one_feature_dic[key],neuron_predicted_groups[i]) 
                #print(f"i {i} neuron_predicted_groups[i] {len(neuron_predicted_groups[i])}")  
                probability_list.append(probability)
            #print(f"key {key} probability_list{probability_list}")
            probability_list = np.array(probability_list)
            onesinglefeatureneuronprobablity[key] = probability_list
           # print(f"key {key} probability_list fuzzy) {probability_list} ")
        return onesinglefeatureneuronprobablity
    

    def getEmbeddingWithClassProbablity_fuzzy(self,X):
        newX =[]
        for x in X:
            newdata =[]
            for j in range(0, len(x)): 
                #**** for certain situation, in the trainng set there is too many data , so we resampled them , as data reslut in the test data the value has but in trainig data it doesnt have
                #print(f"self.all_features_mapping_fuzzy[j].keys() {self.all_features_mapping_fuzzy[j].keys()}")
                if x[j] in self.all_features_mapping_fuzzy[j].keys():
                    #print(f"j {j} x[j] {x[j]}self.all_features_mapping_fuzzy[j]   {self.all_features_mapping_fuzzy[j]}")
                    for value in self.all_features_mapping_fuzzy[j][x[j]]:
                        newdata.append(value) 
                else:
                   # print(f"j{j} x[j]   {x[j] }  self.all_features_mapping[j].keys() {self.all_features_mapping[j].keys()}")
                    fakekey = list(self.all_features_mapping_fuzzy[j])[0]
                    #**** it is not correct, just for a certain dataset, which has lots of data but certain features have very small propration, so when resample the traiing data, that feature is not incluced, but in the test data it has such feature value 
                    for value in self.all_features_mapping_fuzzy[j][fakekey]:
                        newdata.append(value) 
           # print(f"the original discrete data : {x} and proposed encoded data representation: {newdata} ")
            newX.append(newdata)
           
       # print(f"new embedding {newX}")
        #print(f"the original discrete data : {X} and proposed encoded data representation: {newX} ")
        return np.array(newX)
    
 


    def getTrainedClassGroupIndexes(self):
        total_class_num = 0
        group = {}
        for i in  range(0, len(self.train_label_all)):
           if self.train_label_all[i] in group:
            group[self.train_label_all[i]].append(i)
           else:  
               group[self.train_label_all[i]] =[i]
               total_class_num = 1 + total_class_num
        self.class_num = total_class_num
        #print(f"self.class_num {self.class_num}")
        return group

    def get_membership_distribution_based_class(self,x):
        self.converted_x_dic ={}
        sliced_x = np.array_split(x,self.class_num)
        slice_length = len(sliced_x[0])
       # print(f"x  { x} sliced_x {sliced_x}  ") 
        for i in range(0,self.class_num) :   
             real_key = self.getkeyBasedIndex( self.class_index_group_dic, i) 
             for j in range(0, slice_length):
                for slice in sliced_x:               
                    if real_key in self.converted_x_dic:
                        self.converted_x_dic[real_key].append(slice[j])
                    else: self.converted_x_dic[real_key] = [slice[j]]
        
       # print(f"self.converted_x_dic  {self.converted_x_dic }  ") 



    def predict(self,X, centroid_dic):

        labels = np.array([self._find_bmu(x,centroid_dic) for x in X])
        #print(f" labels {labels}")
        return labels
    
    def predictLikeSom(self,X):
        labels=[]
        for x in X:
            bmu_index_sequence_dic = self.find_bmu_sequence_withJSD(x)
          
            #print(f"bmu_index_sequence_dic {bmu_index_sequence_dic} self.getkeyBasedIndex(self, bmu_index_sequence_dic, 0) {self.getkeyBasedIndex(bmu_index_sequence_dic, 0)}")
            labels.append(self.getkeyBasedIndex(bmu_index_sequence_dic, 0))

        return labels
    

    def getEmbeddingWithNeuronProbablity_fuzzy(self,X):
        newX =[]
        for x in X:
            newdata =[]
            for j in range(0, len(x)): 
                #**** for certain situation, in the trainng set there is too many data , so we resampled them , as data reslut in the test data the value has but in trainig data it doesnt have
                #print(f"self.all_features_mapping_fuzzy[j].keys() {self.all_features_mapping_fuzzy[j].keys()}")
                if x[j] in self.all_features_nuron_mapping_fuzzy[j].keys():
                    #print(f"j {j} x[j] {x[j]}self.all_features_mapping_fuzzy[j]   {self.all_features_mapping_fuzzy[j]}")
                    for value in self.all_features_nuron_mapping_fuzzy[j][x[j]]:
                        newdata.append(value) 
                else:
                   # print(f"j{j} x[j]   {x[j] }  self.all_features_mapping[j].keys() {self.all_features_mapping[j].keys()}")
                    fakekey = list(self.all_features_nuron_mapping_fuzzy[j])[0]
                    #**** it is not correct, just for a certain dataset, which has lots of data but certain features have very small propration, so when resample the traiing data, that feature is not incluced, but in the test data it has such feature value 
                    for value in self.all_features_nuron_mapping_fuzzy[j][fakekey]:
                        newdata.append(value) 
            newX.append(newdata)
           
       # print(f"new embedding {newX}")
        #print(f"the original discrete data : {X} and proposed encoded data representation: {newX} ")
        return np.array(newX)
    
    def get_indices_and_data_in_predicted_clusters(self,class_num_predicted,predicted_label):
            
            """
            predicted_label = [1,1,2,3,1,1,2,1]
            idx start from 0 to n
            class_label index also start from 0 to n
            """

            clusters_indexes = []
            #clusters_datas = []
            #print(class_num_predicted)  
            for i in range(0,class_num_predicted):
                newlist = []
                #newdatalist = []
                for idx, y in enumerate(predicted_label): 
                    # is the cluster label
                    if(y == i):
                        x = idx
                        x = int(x)                      
                        newlist.append(x)  
                        #newdatalist.append(data_set[x])                        
                clusters_indexes.append(newlist)
                #clusters_datas.append(np.array(newdatalist))
            #return clusters_indexes,clusters_datas
            #DO NOT NEED clusters_datas TO SAVE ram
            return clusters_indexes   
    def _find_bmu(self,x, centroid_dic):
        """
        Find the index of the best matching unit for the input vector x.
        """  
    
        all_edistance =[]      
        for  key in centroid_dic :
           all_edistance.append(np.linalg.norm(x-centroid_dic[key]))

        return self.getkeyBasedIndex( centroid_dic, np.argmin(all_edistance))

        x_stack = np.stack([x]*(newWeights.shape[0]), axis=0)
        # Calculate distance between x and each weight  ， it use the norm to represent the distance of the concept of vector x_stack - newWeights
       # if showlog:
        #    print("x {} x_stack{}  newWeights {} m {} n{} dim{}".format(x, x_stack, newWeights, self.m,self.n,self.dim))
       # if x_stack.shape != newWeights.shape:
        #    print("x {} x_stack{}  newWeights {} m {} n{} dim{}".format(x, x_stack, newWeights, self.m,self.n,self.dim))
        distance = np.linalg.norm((x_stack - newWeights).astype(float), axis=1)
        # Find index of best matching unit
        return np.argmin(distance)

    def _find_bmu_JSD(self,x, centroid_dic):
        """
        Find the index of the best matching unit for the input vector x.
        """      
        all_jsd =[]      
        for  key in centroid_dic :
           all_jsd.append(self.JSD(x,centroid_dic[key]))

        return self.getkeyBasedIndex( centroid_dic, np.argmin(all_jsd))


    def getkeyBasedIndex(self, dct, n): #don't use dict as  a variable name
        try:
            #print(f"n {n} list(dct)[n] {list(dct)[n]}")
            return list(dct)[n] # or sorted(dct)[n] if you want the keys to be sorted
        except IndexError:
            print ('not enough keys')

    def JSD(self,P, Q):   
            _P = P / norm(P, ord=1)
            _Q = Q/ norm(Q, ord=1)
            _M = 0.5 * (_P + _Q)
            #print(f"Q {Q}")
            return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


    def getCentroidsByMean(self):
        self.centroid_dic = {}
        for key in self.class_index_group_dic:
            total_sum =[]
            for item  in  self.class_index_group_dic[key]:
                total_sum.append( self.train_new_embedding_fcg_fuzzy[item])
                #print(f" item {item } self.train_new_embedding_fcg_fuzzy[item] {self.train_new_embedding_fcg_fuzzy[item] }")
            #print(f" total_sum {total_sum}")
            self.centroid_dic[key] = np.mean(total_sum, axis=0)

    def find_bmu_sequence_withJSD(self, x):
        self.get_membership_distribution_based_class(x)
        all_jsd_dic ={}      
      
        for  key in self.centroid_dic :
           # print(f" self.centroid_dic  { self.centroid_dic }") 
            all_jsd_dic[key] = self.JSD(self.converted_x_dic[key],self.centroid_dic[key])
       # print(f" self. all_jsd_dic  {  all_jsd_dic }") 
        sorted_bmu_index_dic = sorted(all_jsd_dic.items(), key=lambda x:x[1])
       # print(f" dict(sorted_bmu_index_dic) { dict(sorted_bmu_index_dic)}")
        return dict(sorted_bmu_index_dic)
    def topology_som(self, som_num):
            start = int(np.sqrt(som_num))
            factor = som_num / start
            while not self.is_integer(factor):
                start += 1
                factor = som_num / start
            return int(factor), start
    
    def is_integer(self,number):
            if int(number) == number:
                return True
            else:
                return False

    def _updateDeltaCentroid_dic(self, x, sorted_bmu_index_dic):
        self.get_membership_distribution_based_class(x)
        #update_ratio_centroid_dic ={}
        #print(f"sorted_bmu_index_dic {sorted_bmu_index_dic}")
        topology_distance = 1
        for key in  sorted_bmu_index_dic:
            #print(f"self.centroid_dic[key] {self.centroid_dic[key]}" )
            #print(f"key {key}  self.centroid_dic[key] before  { self.centroid_dic[key] }")
           # print(f"key {key} sorted_bmu_index_dic[key] {sorted_bmu_index_dic[key]}")
            neighbourhood = self.lr * math.exp(-topology_distance*self.topology_distance_ratio/(self.sigma ** 2))   
            #print(f"sliced_x_dic[key]  {self.converted_x_dic[key] } self.centroid_dic[key] {self.centroid_dic[key]}" )
            substract_list = list(map(lambda a, b: a - b, self.converted_x_dic[key], self.centroid_dic[key]))
          #  print(f"substract_list {substract_list} neighbourhood {neighbourhood}" )
            substract_list = [i * neighbourhood for i in substract_list]
           # print(f"substract_list2 {substract_list}" )
            self.centroid_dic[key] = list(map(lambda a, b: a + b, self.centroid_dic[key], substract_list))
            #print(f"self.centroid_dic[key] {self.centroid_dic[key]}" )
            #print(f"key {key}  update_delta_centroid_dic[key]  {update_ratio_centroid_dic[key] } sorted_bmu_index_dic[key] {sorted_bmu_index_dic[key]}")
           # self.centroid_dic[key] =[value +  update_ratio_centroid_dic[key] for value in self.centroid_dic[key]]
            #print(f"key {key}  self.centroid_dic[key]  { self.centroid_dic[key] }")
            topology_distance = topology_distance+1
        #print(f"  self.centroid_dic { self.centroid_dic}")
    def _updateDeltaCentroid_dic2(self,x, sorted_bmu_index_dic):
            for key in  sorted_bmu_index_dic:
                print(f"self.centroid_dic[key] 1 {self.centroid_dic[key]}")
                print(f"x {x}")
                self.centroid_dic[key] =[(g + h) / 2 for g, h in zip(self.centroid_dic[key], x)]
                print(f"self.centroid_dic[key] {self.centroid_dic[key]}")
                return
                    

    def getCentroidsByUpdate(self,X, manually_choose = False):
        #___initialize centroid _____
        
        if  manually_choose == True: 
            #manually_choose_dic ={3:69,4:57,0:91,1:110,2:105}
             #manually_choose_dic ={0:115,4:91,3:99,2:47,1:117}
            manually_choose_dic ={0:148,1:21,2:1,3:109,4:39} #drug
            #manually_choose_dic ={0:80,2:136,1:26,3:165} #student  
          # manually_choose_dic ={1:1135,2:157,0:12,3:235}
          #  manually_choose_dic ={0:309,1:1113,2:219} # netflex
           # manually_choose_dic ={0:164,1:314}    #hair 
            #manually_choose_dic ={3:257,2:245,5:630,1:400,4:16,7:522,6:409,8:31,9:686}     #Average Time         
        self.centroid_dic = {}
        random_dic ={}
        for key in self.class_index_group_dic: 
            if manually_choose == False:     
                choosen_index = random.choice(self.class_index_group_dic[key])
                random_dic[key]=choosen_index
                random_controid = self.train_new_embedding_fcg_fuzzy[choosen_index]
              #  print(f"random_controid {random_controid}  manually_choose_dic[key] {manually_choose_dic[key]}")
            else: random_controid = self.train_new_embedding_fcg_fuzzy[manually_choose_dic[key]]
            self.get_membership_distribution_based_class(random_controid)
            self.centroid_dic[key] = self.converted_x_dic[key]
        print(f"self.random_dic {random_dic}")
       # self.getCnetroidByGruopMean(X)
        #___update cenroid_____
        for x in X:
            bmu_index_sequence_dic = self.find_bmu_sequence_withJSD(x)
            self._updateDeltaCentroid_dic(x, bmu_index_sequence_dic)
            #print(f"updated self.centroid_dic {self.centroid_dic}")

    def getCnetroidByGruopMean(self,X):
        self.centroid_dic = {}
        for key in self.class_index_group_dic:
            total =[]
            for index in self.class_index_group_dic[key]:
                total.append(X[index])
            self.centroid_dic[key] =np.mean(total, axis=0)


    def do_FCG(self, topology_distance_ratio = 2, lr = 1, sigma=1):
    
        self.lr = lr 
        self.sigma = sigma
        self.topology_distance_ratio = topology_distance_ratio

        self.all_feature_groups = self.getAllfeatureGroups() 

        self.class_index_group_dic = self.getTrainedClassGroupIndexes() #also get self.class_num
        #print(f"self.class_index_group {self.class_index_group_dic}")
  
        
        self.getEachClassProbabilityOfEachFeatureValue_fuzzy(self.class_index_group_dic)

       
        self.train_new_embedding_fcg_fuzzy = self.getEmbeddingWithClassProbablity_fuzzy(self.data_train_discrete_unnormalized)  

       # print(f"self.train_new_embedding_fcg_fuzzy {self.train_new_embedding_fcg_fuzzy}")
       
        # get centeried of Fuzzy Memebership Distribution in each class
        #self.getCentroidsByUpdate(  self.train_new_embedding_fcg_fuzzy,False )
        self.getCentroidsByUpdate(  self.train_new_embedding_fcg_fuzzy,False) #False
        #self.getCnetroidByGruopMean(  self.train_new_embedding_fcg_fuzzy )
        
       # print(f"centeroid_dic {self.centeroid_dic}")
        # ______________finish training ________________
        self.test_new_embedding_fcg_fuzzy = self.getEmbeddingWithClassProbablity_fuzzy(self.data_test_discrete_unnormalized)  
        #print(f"self.test_new_embedding_fcg_fuzzy  {self.test_new_embedding_fcg_fuzzy }")
        # predict

        predict_label = self.predictLikeSom(self.test_new_embedding_fcg_fuzzy)
       
       # print(f"predict_label  {predict_label }")
        # baseline
        clf= GradientBoostingClassifier()
    #    clf = MLPClassifier(solver= 'sgd',activation = 'tanh',learning_rate ='adaptive',hidden_layer_sizes =(2000,10)  )
        #clf= LogisticRegression()
        #clf = TabNetClassifier() 
        #clf = LinearDiscriminantAnalysis(solver ='eigen')
       # clf = RandomForestClassifier()
       #clf = KNeighborsClassifier(leaf_size = 40)
        #clf = svm.SVC()
        #clf = GaussianNB(var_smoothing = 0.09)
        clf = DecisionTreeClassifier(min_samples_split = 30, min_samples_leaf =25,max_features = 3 )
        clf.fit(self.data_train_discrete_unnormalized, self.train_label_all)

        class_result_test = clf.predict(self.data_test_discrete_unnormalized)
        self.getScores("Baseline",self.test_label_all,class_result_test)
        
        self.getScores("FCG",self.test_label_all, predict_label)
        #self.getScores("FCG",self.test_label_all, predict_label)

        if self.accuracy_score_fcg < self.accuracy_score_baseline:
             print("Not good accuracy result !!!!!")
        if self.recall_score_fcg < self.recall_score_baseline:
             print("Not good recall_score result !!!!!")
        if self.precision_score_fcg < self.precision_score_baseline:
            print("Not good precision_score_ !!!!!")
        if self.f1_score_fcg < self.f1_score_baseline:
             print("Not good accuracy result !!!!!")
        return
        
        
        #########************ SOG
        
        m, n = self.topology_som(self.class_num)
        self.som_discrete = newSom.SOM(m , n, dim= np.shape(self.data_train_discrete_unnormalized)[1]) 

        self.som_discrete.fit(self.data_train_discrete_unnormalized)   
     #   print(f"elf.som.weights0 shape {self.som.weights0.shape}")
        weight_original = self.som_discrete.weights0
        self.train_W0_predicted_label = self.som_discrete.predict(self.data_train_discrete_unnormalized,weight_original)  

        predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som_discrete.weights0.shape[0], self.train_W0_predicted_label) 
        self.getEachNeuronProbabilityOfEachFeatureValue_fuzzy(predicted_clusters_indexes)
        self.train_new_embedding_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized)
        self.test_new_embedding_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized)
       

        #print(f"self.test_new_embedding_sog_fuzzy  {self.test_new_embedding_sog_fuzzy }")
        

        # baseline
       # clf= GradientBoostingClassifier()
        #clf= LogisticRegression()
        clf = KNeighborsClassifier()
        #clf = RandomForestClassifier()
        #clf = KNeighborsClassifier()
        #clf = svm.SVC()
        #clf = GaussianNB()
        #clf = DecisionTreeClassifier()
        clf.fit(self.train_new_embedding_sog_fuzzy, self.train_label_all)
        class_result_test = clf.predict(self.test_new_embedding_sog_fuzzy)


        self.getScores("Baseline",self.test_label_all,class_result_test)
        
        self.getScores("FCG",self.test_label_all, predict_label)
        #self.getScores("FCG",self.test_label_all, predict_label)

        if self.accuracy_score_fcg < self.accuracy_score_baseline:
             print("Not good accuracy result !!!!!")
        if self.recall_score_fcg < self.recall_score_baseline:
             print("Not good recall_score result !!!!!")
        if self.precision_score_fcg < self.precision_score_baseline:
            print("Not good precision_score_ !!!!!")
        if self.f1_score_fcg < self.f1_score_baseline:
             print("Not good accuracy result !!!!!")


     

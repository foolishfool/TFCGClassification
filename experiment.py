#!/usr/bin/env python
# coding: utf-8

from pickle import TRUE
import matplotlib.pyplot as plt
from typing import List
from scipy import stats
import FCG
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import researchpy as rp
from sklearn.model_selection import KFold

class Experiment():

        def __init__(self):
         return

        def __defaults__(self):
         return
        # smooth the curve


        def InitializedExperimentDataList(self,
                                        nfolder,
                                        dataread,
                                        test_score_baseline_accuracy,
                                        test_score_baseline_recall,
                                        test_score_baseline_precision,
                                        test_score_baseline_f1,
                                        test_score_fcg_accuracy,
                                        test_score_fcg_recall,
                                        test_score_fcg_precision,
                                        test_score_fcg_f1,
                                        topology_ratio,
                                        max_accuracy_score_fcg,
                                        max_recall_score_fcg,
                                        max_precision_score_fcg,
                                        max_f1_score_fcg
                                      ):

            if nfolder !=0:              
              kfold = KFold(nfolder)
            #  KFold(n_splits=nfolder, random_state=None, shuffle=False)
              #for i, (train_index, test_index) in enumerate(kfold.split(dataread)):
              accuracy_score_baseline =[]
              recall_score_baseline =[]
              precision_score_baseline =[]
              f1_score_baseline = []
              accuracy_score_fcg =[]
              recall_score_fcg =[]
              precision_score_fcg =[]
              f1_score_fcg = []
              
     
              k = 0
              for train, test in kfold.split(dataread.all_data):
                  # print(f" dataread.all_data { dataread.all_data.shape}")
                   Comparision = FCG.FCG(
                        dataread.data_train,
                        dataread.data_train_continuous, 
                        dataread.all_data_discrete[train,:],
                        dataread.data_test,
                        dataread.data_test_continuous,
                        dataread.all_data_discrete[test,:],
                        dataread.label_all[train],
                        dataread.label_all[test],topology_ratio)     

                   Comparision.do_FCG(topology_ratio)
                   accuracy_score_baseline.append(Comparision.accuracy_score_baseline)
                   #print(f"accuracy_score_baseline {k} {Comparision.accuracy_score_baseline}")
                   recall_score_baseline.append(Comparision.recall_score_baseline)
                   precision_score_baseline.append(Comparision.precision_score_baseline)
                   f1_score_baseline.append(Comparision.f1_score_baseline)
                  
                #   print(f"accuracy_score_fcg1 {k} {Comparision.accuracy_score_fcg}")
                   accuracy_score_fcg.append(Comparision.accuracy_score_fcg)
                   if (max_accuracy_score_fcg <Comparision.accuracy_score_fcg):
                     max_accuracy_score_fcg =Comparision.accuracy_score_fcg
                   recall_score_fcg.append(Comparision.recall_score_fcg)
                   if (max_recall_score_fcg <Comparision.recall_score_fcg):
                     max_recall_score_fcg =Comparision.recall_score_fcg
                   precision_score_fcg.append(Comparision.precision_score_fcg)
                   if (max_precision_score_fcg <Comparision.precision_score_fcg):
                     max_precision_score_fcg =Comparision.precision_score_fcg
                   f1_score_fcg.append(Comparision.f1_score_fcg)
                   if (max_f1_score_fcg <Comparision.f1_score_fcg):
                     max_f1_score_fcg =Comparision.f1_score_fcg
                   k =k +1
                
              test_score_baseline_accuracy.append(np.mean(accuracy_score_baseline))
              print(f"np.mean(accuracy_score_baseline)  ------------------------------- {np.mean(accuracy_score_baseline)}")
              test_score_baseline_recall.append(np.mean(recall_score_baseline))
              test_score_baseline_precision.append(np.mean(precision_score_baseline))
              test_score_baseline_f1.append(np.mean(f1_score_baseline))

              test_score_fcg_accuracy.append(np.mean(accuracy_score_fcg))
              print(f"np.mean(accuracy_score_fcg) ------------------------------- {np.mean(accuracy_score_fcg)}")
              test_score_fcg_recall.append(np.mean(recall_score_fcg))
              test_score_fcg_precision.append(np.mean(precision_score_fcg))
              test_score_fcg_f1.append(np.mean(f1_score_fcg) )
                
              return max_accuracy_score_fcg,max_recall_score_fcg,max_precision_score_fcg,max_f1_score_fcg
                           
            else:  
            
                Comparision = FCG.FCG(
                        dataread.data_train,
                        dataread.data_train_continuous,
                        dataread.data_train_discrete,
                        dataread.data_test,
                        dataread.data_test_continuous,
                        dataread.data_test_discrete,
                        dataread.label_train,
                        dataread.label_test,topology_ratio)                        


                Comparision.do_FCG(topology_ratio)

                self.all_features_mapping_fuzzy =Comparision.all_features_mapping_fuzzy
                self.train_new_embedding_fcg_fuzzy =Comparision.train_new_embedding_fcg_fuzzy
                self.class_index_group_dic =Comparision.class_index_group_dic
                self.class_num =Comparision.class_num
                self.test_new_embedding_fcg_fuzzy =Comparision.test_new_embedding_fcg_fuzzy
                test_score_baseline_accuracy.append(Comparision.accuracy_score_baseline)
                test_score_baseline_recall.append(Comparision.recall_score_baseline)
                test_score_baseline_precision.append(Comparision.precision_score_baseline)
                test_score_baseline_f1.append(Comparision.f1_score_baseline)

                test_score_fcg_accuracy.append(Comparision.accuracy_score_fcg)
                test_score_fcg_recall.append(Comparision.recall_score_fcg)
                test_score_fcg_precision.append(Comparision.precision_score_fcg)
                test_score_fcg_f1.append(Comparision.f1_score_fcg)
                
                
               
                if (max_accuracy_score_fcg <Comparision.accuracy_score_fcg):
                     max_accuracy_score_fcg =Comparision.accuracy_score_fcg
                  
                if (max_recall_score_fcg <Comparision.recall_score_fcg):
                     max_recall_score_fcg =Comparision.recall_score_fcg
                
                if (max_precision_score_fcg <Comparision.precision_score_fcg):
                     max_precision_score_fcg =Comparision.precision_score_fcg
         
                if (max_f1_score_fcg <Comparision.f1_score_fcg):
                     max_f1_score_fcg =Comparision.f1_score_fcg
                
                
                return max_accuracy_score_fcg,max_recall_score_fcg,max_precision_score_fcg,max_f1_score_fcg

        def Ttest( self, dataread, scope_num, topology_ratio, nfolder =0):
            
            all_test_score_baseline_accuracy =[]
            all_test_score_baseline_recall =[]
            all_test_score_baseline_precision =[]
            all_test_score_baseline_f1 =[]

            
            all_test_score_fcg_accuracy =[]
            all_test_score_fcg_recall =[]
            all_test_score_fcg_precision =[]
            all_test_score_fcg_f1 =[]

            plot_unit = [1]
            
            max_accuracy_score_fcg =0
            max_recall_score_fcg =0
            max_precision_score_fcg =0
            max_f1_score_fcg =0

            y = 1
            while y <= scope_num:
                print("Experiment number: {}".format(y))           
                accuracy_score_fcg,recall_score_fcg,precision_score_fcg,f1_score_fcg =self.InitializedExperimentDataList(nfolder,
                                        dataread,
                                        all_test_score_baseline_accuracy,
                                        all_test_score_baseline_recall,
                                        all_test_score_baseline_precision,
                                        all_test_score_baseline_f1,
                                        all_test_score_fcg_accuracy,
                                        all_test_score_fcg_recall,
                                        all_test_score_fcg_precision,   
                                        all_test_score_fcg_f1,
                                        topology_ratio,
                                         max_accuracy_score_fcg,
                                         max_recall_score_fcg,
                                         max_precision_score_fcg,
                                         max_f1_score_fcg                  
                                        )      
                
                
                if (max_accuracy_score_fcg <accuracy_score_fcg):
                     max_accuracy_score_fcg =accuracy_score_fcg
                
                if (max_recall_score_fcg <recall_score_fcg):
                     max_recall_score_fcg =recall_score_fcg
                   
                if (max_precision_score_fcg <precision_score_fcg):
                     max_precision_score_fcg =precision_score_fcg
                   
                if (max_f1_score_fcg <f1_score_fcg):
                     max_f1_score_fcg =f1_score_fcg  
                y =y + 1
                if(y<= scope_num):
                    plot_unit.append(y)
            
            
            
            
            print(f"max_accuracy_score_fcg {max_accuracy_score_fcg}")
            print(f"max_recall_score_fcg {max_recall_score_fcg}")
            print(f"max_precision_score_fcg {max_precision_score_fcg}")
            print(f"max_f1_score_fcg {max_f1_score_fcg}")
               
            figure, axis = plt.subplots(1, 4,figsize =(12, 5))
            axis[0].set_title("Accuracy Score")               
            axis[1].set_title("Recall Score")
            axis[2].set_title("Precision Score") 
            axis[3].set_title("F1 Score")

   
          

            print(f"all_accuracy_score_baseline mean {np.mean(all_test_score_baseline_accuracy)}")
            print(f"all_recall_score_baseline mean {np.mean(all_test_score_baseline_recall)}")
            print(f"all_precision_score_baseline mean {np.mean(all_test_score_baseline_precision)}")
            print(f"all_f1_score_baseline mean {np.mean(all_test_score_baseline_f1)}")

            print(f"all_accuracy_score_fcg mean {np.mean(all_test_score_fcg_accuracy)}")
            print(f"all_recall_score_fcg mean {np.mean(all_test_score_fcg_recall)}")
            print(f"all_precision_score_fcg mean {np.mean(all_test_score_fcg_precision)}")
            print(f"all_f1_score_fcg mean {np.mean(all_test_score_fcg_f1)}")




            axis[0].set_xlabel('Experiment number')
            axis[1].set_xlabel('Experiment number')
            axis[2].set_xlabel('Experiment number')

            axis[3].set_xlabel('Experiment number')
            #axis[4].set_xlabel('Neuron number')

            print(f"len plot_unit {len(plot_unit)}  len (all_test_score_baseline_accuracy) {len(all_test_score_baseline_accuracy)}")
            axis[0].plot(plot_unit,all_test_score_baseline_accuracy,'r',label ='all_accuracy_score_baseline')
            axis[0].plot(plot_unit,all_test_score_fcg_accuracy,'b',label ='all_accuracy_score_fcg')
            axis[0].legend(loc='best')

            axis[1].plot(plot_unit,all_test_score_baseline_recall,'r',label ='all_recall_score_baseline')
            axis[1].plot(plot_unit,all_test_score_fcg_recall,'b',label ='all_recall_score_fcg')
            axis[1].legend(loc='best')


            axis[2].plot(plot_unit,all_test_score_baseline_precision,'r',label ='all_precision_score_baseline')
            axis[2].plot(plot_unit,all_test_score_fcg_precision,'b',label ='all_precision_score_fcg')
            axis[2].legend(loc='best')


            axis[3].plot(plot_unit,all_test_score_baseline_f1,'r',label ='all_f1_score_baseline')
            axis[3].plot(plot_unit,all_test_score_fcg_f1,'b',label ='all_f1_score_fcg')
            axis[3].legend(loc='best')

            
     


            plt.show()
            
            print(f"Normal test + all_test_score_baseline_accuracy")
            shapiro_test = stats.shapiro(all_test_score_baseline_accuracy)
            print(shapiro_test.pvalue)
            print(f"Normal test + all_test_score_fcg_accuracy")
            shapiro_test = stats.shapiro(all_test_score_fcg_accuracy)
            print(shapiro_test.pvalue)

                                                    
                      
            df1 = pd.DataFrame(all_test_score_baseline_accuracy, columns = ['all_accuracy_score_baseline'])
            df2 = pd.DataFrame(all_test_score_fcg_accuracy, columns = ['all_accuracy_score_fcg'])

               
            print("Accuracy Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_accuracy_score_baseline'], group1_name= "all_accuracy_score_baseline",
                                            group2= df2['all_accuracy_score_fcg'], group2_name= "all_accuracy_score_fcg")
            print(summary)
            print(results)


            print(f"Normal test + all_test_score_baseline_recall")
            shapiro_test = stats.shapiro(all_test_score_baseline_recall)
            print(shapiro_test.pvalue)
            print(f"Normal test + all_test_score_fcg_recall")
            shapiro_test = stats.shapiro(all_test_score_fcg_recall)
            print(shapiro_test.pvalue)


            df1 = pd.DataFrame(all_test_score_baseline_recall, columns = ['all_recall_score_baseline'])
            df2 = pd.DataFrame(all_test_score_fcg_recall, columns = ['all_recall_score_fcg'])

               
            print("Recall Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_recall_score_baseline'], group1_name= "all_recall_score_baseline",
                                            group2= df2['all_recall_score_fcg'], group2_name= "all_recall_score_fcg")
            print(summary)
            print(results)

            print(f"Normal test + all_test_score_baseline_precision")
            shapiro_test = stats.shapiro(all_test_score_baseline_precision)
            print(shapiro_test.pvalue)
            print(f"Normal test + all_test_score_fcg_precision")
            shapiro_test = stats.shapiro(all_test_score_fcg_precision)
            print(shapiro_test.pvalue)
            
            df1 = pd.DataFrame(all_test_score_baseline_precision, columns = ['all_precision_score_baseline'])
            df2 = pd.DataFrame(all_test_score_fcg_precision, columns = ['all_precision_score_fcg'])

               
            print("Precision Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_precision_score_baseline'], group1_name= "all_precision_score_baseline",
                                            group2= df2['all_precision_score_fcg'], group2_name= "all_precision_score_fcg")
            print(summary)
            print(results)


            print(f"Normal test + all_test_score_baseline_f1")
            shapiro_test = stats.shapiro(all_test_score_baseline_f1)
            print(shapiro_test.pvalue)
            print(f"Normal test + all_test_score_fcg_f1")
            shapiro_test = stats.shapiro(all_test_score_fcg_f1)
            print(shapiro_test.pvalue)
            
            df1 = pd.DataFrame(all_test_score_baseline_f1, columns = ['all_f1_score_baseline'])
            df2 = pd.DataFrame(all_test_score_fcg_f1, columns = ['all_f1_score_fcg'])

               
            print("F1 Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_f1_score_baseline'], group1_name= "all_f1_score_baseline",
                                            group2= df2['all_f1_score_fcg'], group2_name= "all_f1_score_fcg")
            print(summary)
            print(results)


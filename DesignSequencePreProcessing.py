# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:41:13 2017

@author: Amit_Regmi
"""
import os
import json
import nltk
import gensim
import numpy as np
from gensim import corpora, models, similarities
import pickle
# fix random seed for reproducibility
np.random.seed(7)
#%%
os.chdir('C:\Users\Amit_Regmi\.spyder\Test_Example\LSTM_Recommendation')    
model = gensim.models.Word2Vec.load('series2vec.bin');

file = open('BOMSeries.json')
bom_file = json.load(file)
bom_list = []
for i in range(len(bom_file)):
    unit = bom_file[i]    
    bom_list.append(unit['SeriesCode'])    

tok_comp = []
for unit in bom_list:
    comp = unit.split(',')
    tok_comp.append(comp)  
#%% prepare the dataset of input to output
num_inputs = 100
max_len = 5
tok_x = []
tok_y = []
for token in tok_comp:
    if len(token) > 10:       
        for i in range(num_inputs):
            	QuestionStart = np.random.randint(len(token)-2)
            	QuestionEnd = np.random.randint(QuestionStart, min(QuestionStart+max_len,len(token)-1))
            	sequence_in = token[QuestionStart:QuestionEnd+1]
                  
            	AnswerStart = QuestionEnd+1
            	if AnswerStart == len(token)-1:
                     sequence_out = token[AnswerStart:AnswerStart+1]        
            	else:
                     AnswerEnd = np.random.randint(AnswerStart, min(AnswerStart+max_len,len(token)-1))
                     sequence_out = token[AnswerStart:AnswerEnd+1]  
                    
            	tok_x.append(sequence_in)            	
            	tok_y.append(sequence_out)

#tok_x[len(tok_x)-1:]=[]
#del tok_y[0] 
#%% prepare the dataset of input to output pairs encoded as vector
vec_end = np.ones((32L,),dtype=np.float32)

vec_x=[]
for module in tok_x:
    module_vec = [model[series] for series in module if series in model.wv.vocab.keys()]
    vec_x.append(module_vec)
    
vec_y=[]
for module in tok_y:
    module_vec = [model[series] for series in module if series in model.wv.vocab.keys()]
    vec_y.append(module_vec)

for token in vec_x:
    token[max_len:]=[]
    token.append(vec_end)
    
for token in vec_x:
    if len(token)<(max_len+1):
        for i in range((max_len+1)-len(token)):
            token.append(vec_end)
            
for token in vec_y:
    token[max_len:]=[]
    token.append(vec_end)
    
for token in vec_y:
    if len(token)<(max_len+1):
        for i in range((max_len+1)-len(token)):
            token.append(vec_end)

            
with open('DesignSequenceQA.pickle','w') as f:
    pickle.dump([vec_x,vec_y],f)            
  #%%          
            
            
            
            
            
            
            

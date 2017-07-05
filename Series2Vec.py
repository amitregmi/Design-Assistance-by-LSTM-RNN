# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 18:22:30 2017

@author: Amit_Regmi
"""

import numpy
import json
import gensim
from gensim import corpora, models, similarities
import pandas as pd
# fix random seed for reproducibility
numpy.random.seed(7)

#%%define the raw dataset
file = open('C:\Users\Amit_Regmi\.spyder\Test_Example\LSTM_Recommendation\BOMSeries.json')
bom_list = json.load(file)
corpus = []
for i in range(len(bom_list)):
    row = bom_list[i]    
    corpus.append(row['SeriesCode'])    

tok_corp = []
for sent in corpus:
    row = sent.split(',')
    tok_corp.append(row)
    
model = gensim.models.Word2Vec(tok_corp, min_count = 1, size = 32)
model.save('series2vec.bin')
#model = gensim.models.Word2Vec.load('series2vec.bin')
#%% get model series name of respective series code
path = 'C:\Users\Amit_Regmi\.spyder\Test_Example\LSTM_Recommendation\series2name.csv'
df = pd.read_csv(path, encoding='utf8')
serieslist = df['series_code'].values.tolist()
seriesname = df['series_name'].values.tolist()
model_seriescode = model.wv.vocab.keys()
model_seriesname = []
import math
for model_series in model.wv.vocab.keys():
    bStatus = 1
    for i in range(len(serieslist)):  
        if(math.isnan(serieslist[i])):   
             continue;
        else:  
            if int(model_series) == int(serieslist[i]):
                model_seriesname.append(seriesname[i])
                bStatus = 0
                break;
    if(bStatus):
        model_seriesname.append(model_series)
#%% getting a list of word vectors. limit to 5000. each is of 32 dimensions
series_vectors = [model[w] for w in model.wv.vocab.keys()]
# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components = 2, verbose = 1, random_state = 0)
tsne_w2v = tsne_model.fit_transform(series_vectors)
# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x','y'])
tsne_df['SeriesName'] = model_seriesname
tsne_df['SeriesCode'] = model_seriescode
#%%
# importing bokeh library for interactive dataviz
import bokeh.plotting as bp 
from bokeh.models import HoverTool, BoxSelectTool, LabelSet
from bokeh.plotting import figure, show, output_notebook, output_file
# defining the chart
output_file("series2vec.html")
output_notebook()
plot_tfidf = figure(plot_width=700, plot_height=600, title="A map of Series vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)
# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
#labels = LabelSet(x='x', y='y', text='SeriesName', level='glyph', x_offset=1, y_offset=1, source=tsne_df, render_mode='canvas')
#plot_tfidf.add_layout(labels)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"SeriesName": "@SeriesName"}
show(plot_tfidf)
#%%


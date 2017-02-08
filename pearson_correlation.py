#!/usr/bin/env python
#src https://www.kaggle.com/arthurtok/d/deepmatrix/imdb-5000-movie-dataset/principal-component-analysis-with-kmeans-visuals

import os, sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.cluster import KMeans # KMeans clustering
from sklearn.preprocessing import StandardScaler #for data standarlisation
import matplotlib.pyplot as plt # Python defacto plotting library
import seaborn as sns # More snazzy plotting library

def remove_NaN_data(data):
    str_list = [] # empty list to contain columns with strings (words)
    for colname, colvalue in data.iteritems():
        if type(colvalue[1]) == str:
             str_list.append(colname)
    # Get to the numeric columns by inversion
    num_list = data.columns.difference(str_list)
    #USe only the numeriv values
    data_num = data[num_list]
    #remove the null values, that is fill NaN with there - FIXME: Rihards, naive implementation
    return data_num.fillna(value=0, axis=1)


def generate_plots(data, non_NaN_data, file_name):
    #plot the hexvalue collerations and pearson colleration
    ##heathmap1 = data.plot(y= 'imdb_score', x ='duration',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Duration')
    ##heathmap2 = data.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=45, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Gross')
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 8))
    plt.title('Pearson Correlation of Features')
    # Draw the heatmap using seaborn
    pearsons = sns.heatmap(non_NaN_data.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black')

    #save plots
    #name = file_name + '_heathmap1.png'
    ##fig = heathmap1.get_figure()
    ##fig.savefig(file_name)

    #name = file_name + '_heathmap2.png'
    ##fig = heathmap2.get_figure()
    ##fig.savefig(file_name)

    name = file_name + '_pearsons.png'
    fig = pearsons.get_figure()
    fig.savefig(name)


def main():

    for infile in sys.argv[1:]:
        file_name, ext = os.path.splitext(infile)
        try:
            if ext != '.csv' :
                raise TypeError('input file should have an .csv extension')
            data = pd.read_csv(infile) #open file
            non_NaN_data = remove_NaN_data(data) #remove non number fields
            generate_plots(data, non_NaN_data, file_name) #generate plots
        except (IOError, TypeError) as err:
            print "Error: cannot process the input file", infile
            print "Error caught: " + repr(err)
            quit(0)

if __name__ == "__main__":
    main()

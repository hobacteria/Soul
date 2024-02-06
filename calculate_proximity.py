import geopandas as gpd
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
bnd_gdf= gpd.read_file('/home/hkkim/Soul/_census_data_2022_4_bnd_dong_bnd_dong_11_2022_2022/bnd_dong_11_2022_2022_4Q.shp')
bnd_gdf = bnd_gdf[['ADM_NM','ADM_CD','geometry']]
amd_dict = dict(zip(bnd_gdf['ADM_CD'] , bnd_gdf['ADM_NM']))
bnd_gdf[['ADM_CD','ADM_NM']].to_csv('/home/hkkim/Soul/merged_Data/ADM_NM.csv',encoding='cp949')
merged_path = '/home/hkkim/Soul/merged_Data'
dates = ['201806','201906','202006','202106','202206']



geo_pd = []
for d in dates:
    geo_tmp = gpd.read_file(merged_path+f'/large_category/{d}/{d}_arregate_LCL.shp')
    geo_tmp['date'] = d
    geo_pd.append(geo_tmp)
    
geo_pd = pd.concat(geo_pd)

geo_pd = geo_pd.groupby(['DL_GD_LCLS','DL_GD_LC_1','REC_LGDNG_','ADM_NM'],as_index = False)['INVC_COUNT'].sum()
product_share = geo_pd.groupby(['DL_GD_LC_1'])['INVC_COUNT'].sum().to_dict()
area_share = geo_pd.groupby(['REC_LGDNG_'])['INVC_COUNT'].sum().to_dict()
total = geo_pd['INVC_COUNT'].sum()

geo_pd['RCA'] = geo_pd.apply(lambda x: (x['INVC_COUNT']/area_share[x['REC_LGDNG_']] ) / (product_share[x['DL_GD_LC_1']] /total ),axis=1)
max(geo_pd['RCA'])


geo_pd['MCP'] = geo_pd.apply(lambda x: 1 if x['RCA'] >= 1 else 0 ,axis=1)

MCP_matrix = geo_pd.pivot_table(index='REC_LGDNG_',columns='DL_GD_LC_1',values='MCP',aggfunc='sum')#.to_csv('/home/hkkim/Soul/merged_Data/large_category/2022_4Q_MCP.csv',encoding='cp949')



def calculate_proximity(MCP_matrix,key_columns : str):
    if key_columns == 'idx':
        MCP_matrix
    elif key_columns == 'col':
        MCP_matrix = MCP_matrix.T
    
    key_columns = MCP_matrix.index
    proximity_matrix = pd.DataFrame(index=key_columns,columns=key_columns)
    
    for i in key_columns:
            
        MCP_i = MCP_matrix.loc[:,MCP_matrix.loc[i] == 1]
        proximity_matrix.loc[:,i] = MCP_i.mean(axis=1)
    
    proximity_matrix_np = proximity_matrix.to_numpy().copy()

    idx = np.triu(proximity_matrix_np) > np.triu(proximity_matrix_np.T)
    proximity_matrix_np[idx] = proximity_matrix_np.T[idx]
    proximity_matrix_np = proximity_matrix_np.T
    
    idx = np.triu(proximity_matrix_np) > np.triu(proximity_matrix_np.T)
    proximity_matrix_np[idx] = proximity_matrix_np.T[idx]
    proximity_matrix_np = proximity_matrix_np.T
    
    proximity_matrix_min = pd.DataFrame(proximity_matrix_np,index=key_columns,columns=key_columns)    
    return proximity_matrix_min
            
#middle_prox = calculate_proximity(MCP_matrix,key_columns='idx')
large_prox = calculate_proximity(MCP_matrix,key_columns='idx')

for i in large_prox.index:
    large_prox.loc[i,i] = 0

## 11210680 관악구
## 11230510 강남구
amd_dict['11230510'] = '강남구 신사동'
amd_dict['11210680'] = '관악구 신사동'

large_prox.index = list(map(lambda x: amd_dict[x],large_prox.index))
large_prox.columns = list(map(lambda x: amd_dict[x],large_prox.columns))
large_prox.to_csv('./large_prox.csv',encoding='cp949')


#prox.to_csv('./prox.csv',encoding='utf-8-sig')


def to_gephi(file_nm,adj):
    
    G = nx.from_pandas_adjacency(adj.astype(float),create_using=nx.Graph)
    ## export to gephi
    nx.write_gexf(G, file_nm)
    
to_gephi('./middle_prox.gexf',middle_prox)

large_prox = pd.read_csv('/home/hkkim/Soul/large_prox.csv')
to_gephi('./large_prox.gexf',large_prox)

large_prox

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='NanumGothicCoding')

plt.figure(figsize=(30, 30))
linked = linkage(large_prox.astype(float), 'single')

dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            labels=large_prox.index,
            show_leaf_counts=True)


large_prox


plt.savefig('dendrogram.png')
plt.clf()
plt.figure(figsize=(10, 10))
plt.hist(large_prox.to_numpy().flatten(),bins = 20)
plt.xlabel('Proximity')
plt.ylabel('Frequency')
plt.savefig('hist.png')
plt.clf()


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import scipy.cluster.hierarchy as sch

import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='NanumGothicCoding')

def plot_corr(df,size=10):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''


    # Compute the correlation matrix for the received dataframe
    corr = df.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdYlGn')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

df_proxi = pd.read_csv("/home/hkkim/Soul/large_prox.csv", index_col=0, encoding='cp949')

# correlation matrix before clustring
plot_corr(df_proxi, size=10)


# correlation matrix after clustring

X = df_proxi.corr().values
d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')
columns = [df_proxi.columns.tolist()[i] for i in list((np.argsort(ind)))]
df_proxi = df_proxi.reindex(columns, axis=1)
plt.clf()
plot_corr(df_proxi, size=100)
plt.savefig('corr.png')
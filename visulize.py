import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shutil
import matplotlib as mpl
import os
from tqdm import tqdm
import seaborn as sns

mpl.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='NanumGothicCoding')

bnd_gdf= gpd.read_file('/home/hkkim/Soul/_census_data_2022_4_bnd_dong_bnd_dong_11_2022_2022/bnd_dong_11_2022_2022_4Q.shp')
bnd_gdf = bnd_gdf[['ADM_NM','ADM_CD','geometry']]

seoul_gdf = gpd.read_file('/home/hkkim/Soul/CJdata_2022/05_TC_NU_SPG_50_METER_grid기준 (2)/TC_NU_SPG_50_METER_11.shp',encoding='cp949')
seoul_gdf = seoul_gdf.to_crs(bnd_gdf.crs)
## 면적 larget category에 사용했다는 것 알려주기 위한 figure

fig, ax = plt.subplots()
seoul_gdf[:2000].plot(color = 'none',edgecolor='red',alpha = 1,figsize=(10,10),ax = ax)
x_range = ax.get_xlim()
y_range = ax.get_ylim()
bnd_gdf.plot(color='white',edgecolor='black',figsize=(10,10),ax = ax)
seoul_gdf[:2000].plot(color = 'none',edgecolor='red',alpha = 1,figsize=(10,10),ax = ax)
plt.xlim(x_range)
plt.ylim(y_range)
plt.xticks([])
plt.yticks([])


plt.savefig('border_line.png')
plt.clf()

fig, ax = plt.subplots()
bnd_gdf.plot(color='white',edgecolor='black',figsize=(10,10),ax = ax)
seoul_gdf[:5000].plot(color = 'white',edgecolor='red',linewidth = 0.1,figsize=(10,10),ax = ax)
plt.savefig('border_line.png')
plt.clf()



## 이후부터는 택배 카테고리에 따른 지도 그리기
merged_path = '/camin1/hkkim/merged_data'
geo_pd_202006_LCL = gpd.read_file(merged_path + '/middle_category/202006/202006_arregate_MCL.shp')

cat_dict = dict(zip(geo_pd_202006_LCL[['DL_GD_LCLS','DL_GD_LC_1']].drop_duplicates()['DL_GD_LCLS'],geo_pd_202006_LCL[['DL_GD_LCLS','DL_GD_LC_1']].drop_duplicates()['DL_GD_LC_1']))


def get_figure_map(geo_pd,cat,key_columns,time,folder_nm):
    
    f = plt.figure(figsize=(40,15))

    ax = f.add_subplot(121)
    geo_pd[geo_pd[key_columns] == cat].plot(column = 'INVC_COUNT',cmap='Blues',legend=True,ax = ax)
    plt.title(f'{time}, seoul aggregation by {cat} category',fontdict={'fontsize': 20})
    plt.xticks([])
    plt.yticks([])
    ## 특수문자 제거    
    plt.xticks([])
    plt.yticks([])
    top_area = geo_pd[geo_pd[key_columns] == cat].sort_values(by='INVC_COUNT',ascending=False)[:3]['ADM_NM']
    
    plt.xlabel(f'top 1 : {top_area.iloc[0]} \n top 2 : {top_area.iloc[1]} \n top 3 : {top_area.iloc[2]}',fontdict={'fontsize': 15})
    ax2 = f.add_subplot(122)
    
    plt.hist(geo_pd[geo_pd[key_columns] == cat]['INVC_COUNT'],bins=100)
    plt.tight_layout()

    cat_title = cat.replace('/',' ')
    if os.path.exists(f'./figs/{folder_nm}/{time}') == False:
        os.makedirs(f'./figs/{folder_nm}/{time}')
        
    plt.savefig(f'./figs/map/{folder_nm}/{time}/{cat_title}.png')
    plt.clf()


    
## figure 폴더에 모으기

## 1.normalize 된 지도
## 1.1 카테고리별 비율
## 1.2 행정동별 비율

## 중분류 수준에서 카테고리별 택배수 보기
## 택배수 많으면 그리기

amd_dict = dict(zip(bnd_gdf['ADM_CD'] , bnd_gdf['ADM_NM']))

dates = ['201806','201906','202006','202106','202206']

for d in tqdm(dates):
    geo_pd = gpd.read_file(merged_path+'/large_category/'+d+'/'+d+'_arregate_LCL.shp')
    geo_pd['ADM_NM'] = geo_pd['REC_LGDNG_'].map(amd_dict)
    
    for cat in geo_pd['DL_GD_LC_1'].unique():
        get_figure_map(geo_pd,cat,'DL_GD_LC_1',d,folder_nm='large_category')

## 동으로 regularize 된 지도
for d in tqdm(dates):
    geo_pd = gpd.read_file(merged_path+'/large_category/'+d+'/'+d+'_arregate_LCL.shp')
    geo_pd['ADM_NM'] = geo_pd['REC_LGDNG_'].map(amd_dict)
    gb_bnd = geo_pd.groupby('ADM_NM',as_index = True)['INVC_COUNT'].sum().to_dict()
    
    for i in range(len(geo_pd)):
        geo_pd.loc[i,'INVC_COUNT'] = geo_pd.loc[i,'INVC_COUNT']/gb_bnd[geo_pd.loc[i,'ADM_NM']]
        
    for cat in geo_pd['DL_GD_LC_1'].unique():
        get_figure_map(geo_pd,cat,'DL_GD_LC_1',d,folder_nm='large_category_regularize_by_dong')

# category별 비율
for d in tqdm(dates):
    geo_pd = gpd.read_file(merged_path+'/large_category/'+d+'/'+d+'_arregate_LCL.shp')
    geo_pd['ADM_NM'] = geo_pd['REC_LGDNG_'].map(amd_dict)
    gb_cat = geo_pd.groupby('DL_GD_LC_1',as_index = True)['INVC_COUNT'].sum().to_dict()

    
    for i in range(len(geo_pd)):
        geo_pd.loc[i,'INVC_COUNT'] = geo_pd.loc[i,'INVC_COUNT']/gb_cat[geo_pd.loc[i,'DL_GD_LC_1']]
        
    for cat in geo_pd['DL_GD_LC_1'].unique():
        get_figure_map(geo_pd,cat,'DL_GD_LC_1',d,folder_nm='large_category_ratio')

## 인구수로 normalize 된 지도
d = '202006'

pop = pd.read_csv('./주민등록인구(연령별_동별)_20240204170058.csv',skiprows=3,na_values='-')
pop.columns = ['gu','dong','total','201806','201906','202006','202106','202206']
pop = pop[pop['dong'] != '소계']
pop[pop['dong'] == '상일동']
pop[pop['dong'] == '상일1동']
pop[pop['dong'] == '상일2동']
pop[pop['dong'] == '강일동']


for d in tqdm(dates):
    geo_pd = gpd.read_file(merged_path+'/large_category/'+d+'/'+d+'_arregate_LCL.shp')
    len(geo_pd['ADM_NM'].unique())
    
    for cat in geo_pd['DL_GD_LC_1'].unique():
        get_figure_map(geo_pd,cat,'DL_GD_LC_1',d,folder_nm='large_category')
set(geo_pd['ADM_NM'].unique()) - set(pop['dong'].unique())








geo_pd = []
for d in dates:
    geo_tmp = gpd.read_file(merged_path+f'/middle_category/{d}/{d}_arregate_MCL.shp')
    geo_tmp['date'] = d
    geo_pd.append(geo_tmp)
    
geo_pd = pd.concat(geo_pd)
geo_pd_cats = geo_pd.groupby(['DL_GD_MCLS','DL_GD_MC_1','date'],as_index = False)['INVC_COUNT'].sum()
len(geo_pd_cats['DL_GD_MCLS'].unique())
for d in dates:
    
    geo_pd_cats[geo_pd_cats['date'] == d].sort_values(by='INVC_COUNT',ascending=False)[:30].plot.bar(x='DL_GD_MCLS',y='INVC_COUNT',figsize=(15,10))
    plt.title(f'{d} top 30 middle category',fontdict={'fontsize': 20})
    plt.xticks(rotation=45,fontsize=15)
    if os.path.exists(f'./figs/middle_category_bar/') == False:
        os.makedirs(f'./figs/middle_category_bar/')
    plt.xlabel('middle category',fontdict={'fontsize': 15})
    plt.tight_layout()
    plt.savefig(f'./figs/middle_category_bar/{d}_top_30_middle_category.png')
    plt.clf()

geo_pd_cats = geo_pd.groupby(['DL_GD_MCLS'],as_index = False)['INVC_COUNT'].sum()
geo_pd_cats.sort_values(by='INVC_COUNT',ascending=False)[:30]




"""living_grid_merged_df[living_grid_merged_df['SPG_INNB'].duplicated()]
living_grid_merged_df[living_grid_merged_df['SPG_INNB'] == '1150000008045200'][['index_left','CREATE_DAT','DGM_NM','PRESENT_SN','geometry']]

fig, ax = plt.subplots()
living_grid_merged_df[living_grid_merged_df['SPG_INNB'] == '1150000008045200']['DGM_NM']
x_range = ax.get_xlim()
y_range = ax.get_ylim()
living_place.iloc[[4023],:].plot(color = 'None',edgecolor = 'black',ax = ax)
living_place.iloc[[1574],:].plot(color = 'None',edgecolor = 'red',ax = ax)
#plt.xlim(x_range)
#plt.ylim(y_range)

plt.savefig('/home/hkkim/Soul/plot2.png')
plt.clf()
"""
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle

# 행정동 데이터 불러오기
bnd_gdf= gpd.read_file('/home/hkkim/Soul/_census_data_2022_4_bnd_dong_bnd_dong_11_2022_2022/bnd_dong_11_2022_2022_4Q.shp')
bnd_gdf = bnd_gdf[['ADM_NM','ADM_CD','geometry']]
## 격자 데이터 불러오기
## shp 확장자만 불러오기
#
#shp_file = os.listdir('/home/hkkim/Soul/CJdata_2022/05_TC_NU_SPG_50_METER_grid기준 (2)')
#
#seoul_gdf = gpd.read_file('/home/hkkim/Soul/CJdata_2022/05_TC_NU_SPG_50_METER_grid기준 (2)/TC_NU_SPG_50_METER_26.shp',encoding='cp949')
#seoul_gdf = seoul_gdf.to_crs(bnd_gdf.crs)
#
#merged_df = gpd.sjoin(bnd_gdf,seoul_gdf,how='right',op='intersects').reset_index(drop=True) # 269336 rows
#
#def filltering_grid(merged_df):
#    duplicated_grid = merged_df.loc[merged_df['SPG_INNB'].duplicated(),'SPG_INNB'].unique()
#    
#    for grid in tqdm(duplicated_grid):
#        unions = merged_df[merged_df['SPG_INNB'] == grid]
#        grid_polygon = unions[['SPG_INNB','geometry']].drop_duplicates()
#        dong_polygon = bnd_gdf.iloc[unions['index_left']]
#        inter_section = gpd.overlay(grid_polygon,dong_polygon,how='intersection')
#        drop_intersections = np.where(inter_section.area != inter_section.area.max())[0]
#        drop_index = unions.iloc[drop_intersections].index
#        
#        merged_df = merged_df.drop(drop_index)    
#    
#    return merged_df
#merged_df = filltering_grid(merged_df)
#
#with open('/home/hkkim/Soul/merged_df.pkl','wb') as f:
#    pickle.dump(merged_df,f)

with open('/home/hkkim/Soul/merged_df.pkl','rb') as f:
    merged_df = pickle.load(f)

#최종 데이터는 shp형식으로, 연도별 택배 종류, 행정동에 대해 aggrigation 대분류, 중분류, 소분류로 나누어서 저장
cj_od_matrix = pd.read_csv('/home/hkkim/Soul/CJdata_2022/INHA_OD_REC_SEOUL_221110.csv',encoding='cp949')
cj_od_matrix.shape #(103342186, 15)
## 중복된 격자 있는지 확인
merged_df['SPG_INNB'].duplicated().sum() # 0
## na, 서울 경계를 벗어난 격자 제거
merged_df_drop_na = merged_df.dropna(subset=['ADM_NM'])
## 격자-행정동 코드 dict
grid2dong_cd = merged_df_drop_na[['ADM_CD','SPG_INNB']].set_index('SPG_INNB').to_dict()
grid2dong_cd = grid2dong_cd['ADM_CD']
## 행정동 코드-행정동 이름 dict
dong_cd2nm = merged_df_drop_na[['ADM_NM','ADM_CD']].drop_duplicates().set_index('ADM_CD').to_dict()



cj_od_matrix['REC_LGDNG_CD'] = cj_od_matrix['REC_SPG_INNB'].apply(lambda x: grid2dong_cd[str(x)])
cj_od_matrix['REC_LGDNG_NM'] = cj_od_matrix['REC_LGDNG_CD'].apply(lambda x: dong_cd2nm['ADM_NM'][x])

cj_od_matrix_gpd = pd.merge(cj_od_matrix,bnd_gdf,left_on='REC_LGDNG_CD',right_on = 'ADM_CD',how='left').drop(['ADM_CD','ADM_NM'],axis=1)


del merged_df_drop_na,merged_df,cj_od_matrix #for memory

#['DL_YM', 'SEND_CTPV_NM', 'SEND_CTPV_CD', 'SEND_CTGG_NM', 'SEND_CTGG_CD','SEND_EMD_NM', 'SEND_LGDNG_CD', 'REC_SPG_INNB', 
# 'DL_GD_LCLS_NM','DL_GD_LCLS_CD', 'DL_GD_MCLS_NM', 'DL_GD_MCLS_CD', 'DL_GD_SCLS_NM','DL_GD_SCLS_CD', 'INVC_COUNT', 'REC_LGDNG_CD', 'REC_LGDNG_NM','geometry']
dates = cj_od_matrix_gpd['DL_YM'].unique()


## only_LCL
#for d in dates:
#    tmp_gpd = cj_od_matrix_gpd.loc[cj_od_matrix_gpd['DL_YM'] == d,['DL_YM','DL_GD_LCLS_NM','DL_GD_LCLS_CD','INVC_COUNT','REC_LGDNG_CD','REC_LGDNG_NM','geometry']].groupby(['REC_LGDNG_CD','DL_GD_LCLS_CD','DL_GD_LCLS_NM','geometry'],as_index = False)['INVC_COUNT'].sum()
#    tmp_gpd = gpd.GeoDataFrame(tmp_gpd,geometry='geometry',crs = bnd_gdf.crs)
#    tmp_gpd.to_file(f'/camin1/hkkim/merged_data/large_category/{d}/{d}_arregate_LCL.shp',encoding='cp949')
#
## MCL by dates
#for d in dates:
#    tmp_gpd = cj_od_matrix_gpd.loc[cj_od_matrix_gpd['DL_YM'] == d,['DL_YM','DL_GD_MCLS_NM', 'DL_GD_MCLS_CD','INVC_COUNT','REC_LGDNG_CD','REC_LGDNG_NM','geometry']].groupby(['REC_LGDNG_CD','DL_GD_MCLS_NM', 'DL_GD_MCLS_CD','geometry'],as_index = False)['INVC_COUNT'].sum()
#    tmp_gpd = gpd.GeoDataFrame(tmp_gpd,geometry='geometry',crs = bnd_gdf.crs)
#    tmp_gpd.to_file(f'/camin1/hkkim/merged_data/middle_category/{d}/{d}_arregate_MCL.shp',encoding='cp949')
#    
    
for d in dates:
    tmp_gpd = cj_od_matrix_gpd.loc[cj_od_matrix_gpd['DL_YM'] == d,['DL_YM','DL_GD_SCLS_NM', 'DL_GD_SCLS_CD','INVC_COUNT','REC_LGDNG_CD','REC_LGDNG_NM','geometry']].groupby(['REC_LGDNG_CD','DL_GD_SCLS_NM', 'DL_GD_SCLS_CD','geometry'],as_index = False)['INVC_COUNT'].sum()
    tmp_gpd = gpd.GeoDataFrame(tmp_gpd,geometry='geometry',crs = bnd_gdf.crs)
    tmp_gpd.to_file(f'/camin1/hkkim/merged_data/small_category/{d}/{d}_arregate_SCL.shp',encoding='cp949')
    
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from multiprocessing import cpu_count
from multiprocessing.sharedctypes import Value, Array
import pickle

def parallelize_dataframe(df,dicts, func):
    df_split = np.array_split(df, 50)
    pool = Pool(50)
    df = pd.concat(pool.map(func, [(df_split[i],i,dicts) for i in range(50)]))
    pool.close()
    pool.join()
    return df

def fill_data(args):
    df = args[0]
    i = args[1]
    dicts = args[2]
    grid2dong_cd = dicts[0]
    grid2dong_cd['ADM_CD_202206'] = grid2dong_cd['ADM_CD'].copy()
    dong_cd2nm = dicts[1]
    pop_dict = dicts[2]
    grid_num_dict = dicts[3]

    grid_set = set(grid2dong_cd['ADM_CD']) | set(grid2dong_cd['ADM_CD_201806']) | set(grid2dong_cd['ADM_CD_201906']) | set(grid2dong_cd['ADM_CD_202006']) | set(grid2dong_cd['ADM_CD_202106'])            
    pop_set = set(pop_dict['201806']) | set(pop_dict['201906']) | set(pop_dict['202006']) | set(pop_dict['202106']) | set(pop_dict['202206'])            
    dong_set = set(dong_cd2nm.keys())
    try:
        df['REC_LGDNG_CD_prev'] = df.apply(lambda x : grid2dong_cd['ADM_CD_' + str(x['DL_YM'])][str(x['REC_SPG_INNB'])] if str(x['REC_SPG_INNB']) in grid_set else np.nan,axis = 1)
        df['REC_LGDNG_CD'] = df.apply(lambda x : grid2dong_cd['ADM_CD'][str(x['REC_SPG_INNB'])] if str(x['REC_SPG_INNB']) in grid_set else np.nan,axis = 1)
        df['REC_LGDNG_NM'] = df.apply(lambda x : dong_cd2nm[str(x['REC_LGDNG_CD_prev'])] if str(x['REC_LGDNG_CD_prev']) in dong_set else np.nan,axis = 1)
        
        ## 
        df['pop_divid_grid_num'] = df.apply(lambda x: pop_dict[str(x['DL_YM'])][str(x['REC_LGDNG_CD_prev'])] / grid_num_dict[str(x['REC_LGDNG_CD_prev'])] if str(x['REC_LGDNG_CD_prev']) in pop_set else np.nan,axis = 1)
        
        df['INVC_COUNT_normalized'] = df.apply(lambda x: x['INVC_COUNT'] / x['pop_divid_grid_num'] / grid_num_dict[str(x['REC_LGDNG_CD'])] if str(x['REC_LGDNG_CD']) in dong_set else np.nan,axis = 1)
    except:
        print(df[df['pop_divid_grid_num'] == 0].head(30))
        
    print(f'done{i}')
    return df

class merging_sdata():
    def __init__(self) -> None:
        """bnd_gdf : 행정동 데이터
        gird_shp_path : 서울 격자 데이터
        cj_data : CJ 택배 데이터
        living_place : 거주지 데이터
        """
        self.bnd_gdf_path = '/home/hkkim/Soul/_census_data_2022_4_bnd_dong_bnd_dong_11_2022_2022/bnd_dong_11_2022_2022_4Q.shp'
        self.prev_bnd_gdf_path = {'201806':'/home/hkkim/Soul/_census_data_2023_bnd_dong_bnd_dong_11_2018_2023/bnd_dong_11_2018_2018_2Q.shp',
                                  '201906':'/home/hkkim/Soul/_census_data_2023_bnd_dong_bnd_dong_11_2019_2023/bnd_dong_11_2019_2019_2Q.shp',
                                  '202006':'/home/hkkim/Soul/_census_data_2023_bnd_dong_bnd_dong_11_2019_2023/bnd_dong_11_2019_2019_2Q.shp',
                                  '202106':'/home/hkkim/Soul/_census_data_2023_bnd_dong_bnd_dong_11_2020_2023/bnd_dong_11_2020_2020_4Q.shp'}
        
        self.gird_shp_path = '/home/hkkim/Soul/CJdata_2022/05_TC_NU_SPG_50_METER_grid기준 (2)/TC_NU_SPG_50_METER_11.shp'
        self.cj_data_path = '/home/hkkim/Soul/CJdata_2022/INHA_OD_REC_SEOUL_221110.csv'
        self.living_place_path = '/home/hkkim/Soul/UPIS_C_UQ111/UPIS_C_UQ111.shp'
        self.population_path = '/home/hkkim/Soul/clean_pop_data.csv'
        if os.path.exists('./bnd_grid_living_merged_df.pkl'):
            with open('./bnd_grid_living_merged_df.pkl','rb') as f:
                self.init_merged_data = True
        else:
            self.init_merged_data = False

        
        if os.path.exists('./check_file.pkl'):
            with open('./check_file.pkl','rb') as f:
                self.init_check_merged_data = True
        else:
            self.init_check_merged_data = False

        self.crs = None
        
    def matching_crs(self,df_1,df_2):
        """_summary_

        Args:
            df_1 (_type_): change crs to df_2
            df_2 (_type_): key_crs
        """
        ch_df_1 = df_1.to_crs(df_2.crs)
        if self.crs == None:
            self.crs = df_2.crs
        return ch_df_1

    def matching_grid_with_boundary(self,merged_df,boundary_df,grid_id_col):
        """when merge result of grid is duplicated, filltering grid
        """
        duplicated_grid = merged_df.loc[merged_df[grid_id_col].duplicated(),grid_id_col].unique()
        cnt = 0
        for grid in tqdm(duplicated_grid):
            unions = merged_df[merged_df[grid_id_col] == grid]
            grid_polygon = unions[[grid_id_col,'geometry']].drop_duplicates()
            dong_polygon = boundary_df.iloc[unions['index_left']]
            inter_section = gpd.overlay(grid_polygon,dong_polygon,how='intersection')
            drop_intersections = np.where(inter_section.area != inter_section.area.max())[0]
            
            drop_index = unions.iloc[drop_intersections].index
            
            merged_df = merged_df.drop(drop_index)    
        
        return merged_df.reset_index(drop=True)


        
    def aggregate_cj_category(self,cj_data,cat : str,dates : list = [201806,201906,202006,202106,202206]):
        cat_cd = {'LCL':'DL_GD_LCLS_CD','MCL':'DL_GD_MCLS_CD','SCL':'DL_GD_SCLS_CD'}
        cat_nm = {'LCL':'DL_GD_LCLS_NM','MCL':'DL_GD_MCLS_NM','SCL':'DL_GD_SCLS_NM'}
        fl_nm = {'LCL':'large','MCL':'middle','SCL':'small'}
        tmp_gpd_list = []
        for d in dates:
            tmp_gpd = cj_data.loc[cj_data['DL_YM'] == d,['DL_YM',cat_nm[cat],cat_cd[cat],'INVC_COUNT','REC_LGDNG_CD','REC_LGDNG_NM','geometry','pop_divid_grid_num','INVC_COUNT_normalized']].groupby(['DL_YM','REC_LGDNG_NM','REC_LGDNG_CD',cat_cd[cat],cat_nm[cat],'geometry'],as_index = False)[['INVC_COUNT','pop_divid_grid_num','INVC_COUNT_normalized']].sum()
            tmp_gpd = gpd.GeoDataFrame(tmp_gpd,geometry='geometry',crs = self.crs)
            if os.path.exists(f'/camin1/hkkim/merged_data_livingplace/{fl_nm[cat]}_category/{d}') == False:
                os.makedirs(f'/camin1/hkkim/merged_data_livingplace/{fl_nm[cat]}_category/{d}')
            tmp_gpd.to_file(f'/camin1/hkkim/merged_data_livingplace/{fl_nm[cat]}_category/{d}/{d}_arregate_{cat}.shp',encoding='cp949')
            
    def get_merged_data(self):
        bnd_gdf = gpd.read_file(self.bnd_gdf_path)
        bnd_gdf = bnd_gdf[['ADM_NM','ADM_CD','geometry']]
        grid_shp_file = gpd.read_file(self.gird_shp_path,encoding='cp949')
        living_place = gpd.read_file(self.living_place_path,encoding='cp949')
        grid_shp_file = self.matching_crs(grid_shp_file,bnd_gdf)
        living_place = self.matching_crs(living_place,bnd_gdf)
        
        ## 격자를 도시계획 구역에 매칭하고, 주거지역에 해당하는 격자만 추출

        living_grid_merged_df = gpd.sjoin(living_place,grid_shp_file,how='right',op='intersects').reset_index(drop=True)
        living_grid_merged_df = self.matching_grid_with_boundary(living_grid_merged_df,living_place,'SPG_INNB')
        
        living_grid_merged_df = living_grid_merged_df.dropna(subset=['DGM_NM'])
        living_grid_merged_df = living_grid_merged_df[living_grid_merged_df['DGM_NM'].str.contains('주거지역')]
        grid_shp_only_living = living_grid_merged_df[grid_shp_file.columns]
        grid_shp_only_living.drop_duplicates(subset='SPG_INNB',inplace=True)
        
        bnd_grid_merged_df = gpd.sjoin(bnd_gdf,grid_shp_only_living,how='right',op='intersects').reset_index(drop=True) # 269336 rows
        bnd_grid_merged_df = self.matching_grid_with_boundary(bnd_grid_merged_df,bnd_gdf,'SPG_INNB')
        with open('./bnd_grid_living_merged_df.pkl','wb') as f:
            pickle.dump(bnd_grid_merged_df,f)
        return bnd_grid_merged_df

    def get_prev_dong_cd(self,merge_df):
        date = ['201806','201906','202006','202106']
        
        for d in date:
            prev_bnd_gdf = gpd.read_file(self.prev_bnd_gdf_path[d])
            try:
                prev_bnd_gdf = prev_bnd_gdf[['adm_dr_nm','adm_dr_cd','geometry']]
            except:
                try:
                    prev_bnd_gdf = prev_bnd_gdf[['ADM_DR_NM','ADM_DR_CD','geometry']]        
                except:
                    prev_bnd_gdf = prev_bnd_gdf.iloc[:,[2,1,3]]
                                
            prev_bnd_gdf.crs = merge_df.crs
            prev_bnd_gdf.columns = [f'ADM_NM_{d}',f'ADM_CD_{d}','geometry']
            merge_df.drop(columns = 'index_left',inplace=True)
            merge_df = gpd.sjoin(prev_bnd_gdf,merge_df,how='right',op='intersects').reset_index(drop=True) 
            
            merge_df = self.matching_grid_with_boundary(merge_df,prev_bnd_gdf,'SPG_INNB')

        
        
        return merge_df
    
    def process(self):

        if self.init_check_merged_data:

            with open('./check_file.pkl','rb') as f:
                bnd_grid_merged_df = pickle.load(f)      
            
        elif self.init_merged_data:
            print('------------find prev merged data------------')
            print('-------------------load data-----------------')
            with open('./bnd_grid_living_merged_df.pkl','rb') as f:
                bnd_grid_merged_df = pickle.load(f)

            bnd_grid_merged_df = self.get_prev_dong_cd(bnd_grid_merged_df)	
            with open('./check_file.pkl','wb') as f:
                pickle.dump(bnd_grid_merged_df,f)
        else:
            bnd_grid_merged_df = self.get_merged_data()
            bnd_grid_merged_df = self.get_prev_dong_cd(bnd_grid_merged_df)
            with open('./check_file.pkl','wb') as f:
                pickle.dump(bnd_grid_merged_df,f)
        
        bnd_grid_merged_df_drop_na = bnd_grid_merged_df.dropna(subset=['ADM_NM'])
        ## 격자-행정동 코드 dict
        grid2dong_cd = bnd_grid_merged_df_drop_na[['ADM_CD','ADM_CD_201806','ADM_CD_201906','ADM_CD_202006','ADM_CD_202106','SPG_INNB']].set_index('SPG_INNB').to_dict()
        ## 행정동 코드-행정동 이름 dict
        dong_cd2nm = dict(zip(bnd_grid_merged_df_drop_na[['ADM_CD','ADM_CD_201806','ADM_CD_201906','ADM_CD_202006','ADM_CD_202106']].melt()['value'],bnd_grid_merged_df_drop_na[['ADM_NM','ADM_NM_201806','ADM_NM_201906','ADM_NM_202006','ADM_NM_202106']].melt()['value']))
        bnd_grid_merged_df_drop_na.loc[bnd_grid_merged_df_drop_na['ADM_NM'].str.contains('구로'),'ADM_NM'].unique()

        pop_data = pd.read_csv(self.population_path)
        tmp_df = pd.DataFrame.from_dict(dong_cd2nm,orient='index').reset_index()
        tmp_df[0] = tmp_df[0].str.replace('·','.')
        pop_data['cd_raw'] = pop_data['gudong'].map(lambda x: tmp_df.loc[tmp_df.iloc[:,1] == x.split()[1],'index'].values)
        pop_data[['cd1','cd2','cd3','cd4']] = pd.DataFrame.from_dict(pop_data['cd_raw'].to_dict(),orient = 'index')
        pop_data = pop_data.drop(columns = ['cd_raw']).melt(id_vars=['gudong','201806','201906','202006','202106','202206'])
        pop_data.dropna(inplace=True)
        
        grid_num_dict = dict()
        for i in ['ADM_CD','ADM_CD_201806','ADM_CD_201906','ADM_CD_202006','ADM_CD_202106']:
            grid_num_dict.update(bnd_grid_merged_df_drop_na[[i,'SPG_INNB']].groupby(i).count().to_dict()['SPG_INNB'])
                
        pop_dict = pop_data[['201806','201906','202006','202106','202206','value']].set_index('value').to_dict()
        
        dicts = [grid2dong_cd,dong_cd2nm,pop_dict,grid_num_dict]
        
        cj_data = pd.read_csv(self.cj_data_path,encoding='cp949')
        
        print('fill dong_cd and dong_nm to cj data')   


        cj_data = parallelize_dataframe(cj_data,dicts,fill_data)
        
        print('-------fill dong_cd and dong_nm to cj data done--------')
        print(cj_data.head(20))
              
        cj_data.dropna(subset=['REC_LGDNG_CD'],inplace=True)
        bnd_gdf = gpd.read_file(self.bnd_gdf_path)
        cj_data = pd.merge(cj_data,bnd_gdf,left_on='REC_LGDNG_CD',right_on = 'ADM_CD',how='left').drop(['ADM_CD','ADM_NM'],axis=1)
        del bnd_grid_merged_df_drop_na,bnd_grid_merged_df
        self.aggregate_cj_category(cj_data,cat = 'LCL')
        self.aggregate_cj_category(cj_data,cat = 'MCL')
        self.aggregate_cj_category(cj_data,cat = 'SCL')
        
self = merging_sdata()
if __name__ == '__main__':
    ms = merging_sdata()
    ms.process()
    print('done')
        

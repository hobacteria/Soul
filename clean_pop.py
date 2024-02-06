import pandas as pd

pop_data = pd.read_csv('/home/hkkim/Soul/주민등록인구(연령별_동별)_20240204170058.csv',header=0,skiprows=(1,2),na_values='-')
        
pop_data = pop_data[pop_data['동별(2)'] != '소계'].drop(columns  ='항목')
pop_data.columns = ['gu','dong','201806','201906','202006','202106','202206']
pop_data['gudong'] = pop_data['gu'] + ' ' + pop_data['dong']
pop_data = pop_data.drop(columns = ['gu','dong'])
pop_data.fillna(0,inplace=True)
pop_data = pop_data.groupby(by = ['gudong']).sum()
pop_data.to_csv('clean_pop_data.csv')
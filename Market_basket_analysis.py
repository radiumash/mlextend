from pandas.core.base import PandasObject
from pandas.core.dtypes.missing import isna, isnull
from pandas.io.parsers import read_csv
import streamlit as st 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import pickle 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from itertools import permutations


import warnings 
warnings.filterwarnings("ignore")

#title
st.title('Product Cross-Selling Predictor')
image=Image.open('header_image.jpg')
st.image(image,use_column_width=True)
#data=st.file_uploader('Upload The Dataset',type=['csv'])
def main():
    st.subheader('Product cross-selling analysis')
    data=st.file_uploader('Upload The Dataset',type=['csv'])
    #d=pd.read_csv(data, encoding='ISO 8859-1')
    #st.success('Data Successfully Uploaded')
    #d=pd.read_csv(data, encoding='ISO 8859-1')
    #st.write('Raw data:',d.head(10))
    if data is not None:
        d=pd.read_csv(data, encoding='ISO 8859-1')
        st.success('Data Successfully Uploaded')
        st.write('Raw data:',d.head(10))
        #d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
        #d['PRODUCTS'] = d['PRODUCTS'].str.strip()
        #d['RECEIPT'] = d['RECEIPT'].astype('str')
        #d = d[~d['RECEIPT'].str.contains('C')]
        #d['Unique Id']=d['STOCKIEST CODE']+'_'+d['MONTH']
        x=['NATIONAL','DELHI','AMBALA','GHAZIABAD','AHMEDABAD','KOLKATTA','PUNE','GUWAHATI','ERNAKULAM']
        option=st.selectbox('Select The Depot:',x)

        if option=='DELHI':
            d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
            d['PRODUCTS'] = d['PRODUCTS'].str.strip()
            #d['RECEIPT'] = d['RECEIPT'].astype('str')
            #d = d[~d['RECEIPT'].str.contains('C')]
            #st.write(d.dtypes)
            #d['Unique Id']=d['STOCKIEST CODE']+d['MONTH']
            #d['Unique Id']=d['STOCKIEST CODE'].map(str)+'_'.map(str)+d['MONTH'].map(str)
            #d['quantitative_v'] = d['quantitative_v'].astype(float)

            basket = (d[d['DEPOT'] =='DELHI']
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            #d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
            #d['PRODUCTS'] = d['PRODUCTS'].str.strip()
            #d['RECEIPT'] = d['RECEIPT'].astype('str')
            #d = d[~d['RECEIPT'].str.contains('C')]
            #d['Unique Id']=d['STOCKIEST CODE']+'_'+d['MONTH']


            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            #st.write(frequent_itemsets.head())
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            #rules["consequents"] = rules["consequents"].astype(str)
            #rules['antecedents']=rules['antecedents'].str.replace('frozenset','')
            #rules['consequents']=rules['consequents'].str.replace('frozenset','')
            #rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            #rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(rules)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #rules["antecedents"] = rules["antecedents"].astype(str)

            #t=rules.rename(columns={'consequents':'R'})
            #st.write(t.head())
            #data2=st.file_uploader('Upload The Products',type=['csv'])
            #if data2 is not None:
                #d1=pd.read_csv(data2)
                #st.success('Data Successfully Uploaded')
                
                #c=d1['Products'] = np.where(d1['Products'] == rules['antecedents'], 'True', 'False')
                #st.write(c)
                    
                
        
            c1=rules['antecedents'].unique()
            cl=list(c1)
            sb=st.selectbox('select product',cl)
            if sb=="Acnemoist Cream 60GM":
                r1=(rules.loc[rules['antecedents']=="Acnemoist Cream 60GM"])
                #st.write(r1)
                #r1x=r1.rename(columns={'consequents':'R'})
                dfr=(r1['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns=['Recommendations']

                #k=k.rename(columns={'0':'Recommendations'})
                #st.write(dfr)
                
                
                #k=pd.DataFrame(dfr)
                st.write(k['Recommendations'].unique())
                #k['Recomendations']=st.write(k['Recomendations'].unique())
                st.write(k.shape)
                #rf1df=pd.DataFrame(rf1)
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write(dx.shape)
                    #st.write(k.info())
                    #st.write(dx.info())
                    
                    #st.write(type(dr))
                    #int_df = pd.merge(k,dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
                    #st.write(pd.merge(k, dx, how='inner', left_on='UniqueID'))
                    #st.write(int_df)
                    #st.write(int_df.shape)
                    
                
                    



                
                
                    
                    
            if sb=='Aknay Bar 100GM':
                r1_2=(rules.loc[rules['antecedents']=='Aknay Bar 100GM'])
                #r1_2x=r1_2.rename(columns={'consequents':'Recommendations'},inplace=False)
                #rf2=st.write(r1_2['Recommendations'].unique())
                dfr=(r1_2['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
                
                #data2=st.file_uploader('Upload The Products',type=['csv'])
                #if data2 is not None:
                    #d1=pd.read_csv(data2)
                    #nc=d1.rename(columns={'Products':'0'},inplace=False)
                    #w=st.write(nc)
                    #st.success('Data Successfully Uploaded')
            if sb=='Androanagen Solution 100Ml':
                r1_3=(rules.loc[rules['antecedents']=='Androanagen Solution 100Ml'])
                #r1_3=r1_3.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_3['Recommendations'].unique())
                dfr=(r1_3['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Tablets 10S':
                r1_4=(rules.loc[rules['antecedents']=='Androanagen Tablets 10S'])
                #r1_4=r1_4.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_4['Recommendations'].unique())
                dfr=(r1_4['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Banatan Cream 50GM':
                r1_5=(rules.loc[rules['antecedents']=='Banatan Cream 50GM'])
                #r1_5=r1_5.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_5['Recommendations'].unique())
                dfr=(r1_5['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Canthex 10 Capsules':
                r1_6=(rules.loc[rules['antecedents']=='Canthex 10 Capsules'])
                #r1_6=r1_6.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_6['Recommendations'].unique())
                dfr=(r1_6['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Es Body Wash 200ML':
                r1_7=(rules.loc[rules['antecedents']=='Es Body Wash 200ML'])
                #r1_7=r1_7.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_7['Recommendations'].unique())
                dfr=(r1_7['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Melawash 100 ML':
                r1_8=(rules.loc[rules['antecedents']=='Melawash 100 ML'])
                #r1_8=r1_8.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_8['Recommendations'].unique())
                dfr=(r1_8['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Triobloc Cream 25GM':
                r1_9=(rules.loc[rules['antecedents']=='Triobloc Cream 25GM'])
                #r1_9=r1_9.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_9['Recommendations'].unique())
                dfr=(r1_9['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Zinikam Cream 200 G':
                r1_10=(rules.loc[rules['antecedents']=='Zinikam Cream 200 G'])
                #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_10['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=="""Nixiyax 15 'S""":
                r1_11=(rules.loc[rules['antecedents']=="""Nixiyax 15 'S"""])
                #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #rf10=st.write(r1_10['Recommendations'].unique())
                dfr=(r1_11['consequents'].head(10))
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Aknay Bar 100GM', 'Androanagen Solution 100Ml'})""":
            #     r1_12=(rules.loc[rules['antecedents']=="""({'Aknay Bar 100GM', 'Androanagen Solution 100Ml'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_12['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Aknay Bar 100GM', 'Acnemoist Cream 60GM'})""":
            #     r1_13=(rules.loc[rules['antecedents']=="""({'Aknay Bar 100GM', 'Acnemoist Cream 60GM'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_13['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Androanagen Solution 100Ml', 'Acnemoist Cream 60GM'})""":
            #     r1_14=(rules.loc[rules['antecedents']=="""({'Androanagen Solution 100Ml', 'Acnemoist Cream 60GM'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_14['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            #     dfr=(r1_14['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Aknay Bar 100GM', 'Androanagen Tablets 10S'})""":
            #     r1_10=(rules.loc[rules['antecedents']=="""({'Aknay Bar 100GM', 'Androanagen Tablets 10S'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_10['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Androanagen Tablets 10S', 'Acnemoist Cream 60GM'})""":
            #     r1_10=(rules.loc[rules['antecedents']=="""({'Androanagen Tablets 10S', 'Acnemoist Cream 60GM'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_10['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Aknay Bar 100GM', 'Banatan Cream 50GM'})""":
            #     r1_10=(rules.loc[rules['antecedents']=="""({'Aknay Bar 100GM', 'Banatan Cream 50GM'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_10['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Acnemoist Cream 60GM', 'Androanagen Tablets 10S'})""":
            #     r1_10=(rules.loc[rules['antecedents']=="""({'Acnemoist Cream 60GM', 'Androanagen Tablets 10S'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_10['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Banatan Cream 50GM', 'Acnemoist Cream 60GM'})""":
            #     r1_10=(rules.loc[rules['antecedents']=="""({'Banatan Cream 50GM', 'Acnemoist Cream 60GM'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_10['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Aknay Bar 100GM', 'Canthex 10 Capsules'})""":
            #     r1_10=(rules.loc[rules['antecedents']=="""({'Aknay Bar 100GM', 'Canthex 10 Capsules'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_10['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Aknay Bar 100GM', 'Melawash 100 ML'})""":
            #     r1_10=(rules.loc[rules['antecedents']=="""({'Aknay Bar 100GM', 'Melawash 100 ML'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_10['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            # if sb=="""({'Acnemoist Cream 60GM', 'Melawash 100 ML'})""":
            #     r1_10=(rules.loc[rules['antecedents']=="""({'Acnemoist Cream 60GM', 'Melawash 100 ML'})"""])
            #     #r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
            #     #rf10=st.write(r1_10['Recommendations'].unique())
            #     dfr=(r1_10['consequents'].head(10))
            #     k=pd.DataFrame(dfr)
            #     k.columns= ['R']
            #     st.write(k)
            #     st.title('Verify Your Plan')
            #     data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
            #     if data3 is not None:
            #         dx=pd.read_csv(data3)
            #         st.success('Data Successfully Uploaded')
            #         st.write(dx)
            #         st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))

            


                
            
                


        
        if option=='NATIONAL':
            basket = (d
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            #st.write(frequent_itemsets.head())
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(rules.head())
            c1=rules['antecedents'].unique()
            cl=list(c1)
            sb=st.selectbox('select product',cl)
            if sb=='Acnemoist Cream 60GM':
                r1=(rules.loc[rules['antecedents']=='Acnemoist Cream 60GM'])
                r1=r1.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1['Recommendations'].unique())
                dfr=(r1['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                

                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))

            if sb=='Aknay Bar 100GM':
                r1_2=(rules.loc[rules['antecedents']=='Aknay Bar 100GM'])
                r1_2=r1_2.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_2['Recommendations'].unique())
                dfr=(r1_2['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))

            if sb=='Androanagen Solution 100Ml':
                r1_3=(rules.loc[rules['antecedents']=='Androanagen Solution 100Ml'])
                r1_3=r1_3.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_3['Recommendations'].unique())
                dfr=(r1_3['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Tablets 10S':
                r1_4=(rules.loc[rules['antecedents']=='Androanagen Tablets 10S'])
                r1_4=r1_4.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_4['Recommendations'].unique())
                dfr=(r1_4['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Banatan Cream 50GM':
                r1_5=(rules.loc[rules['antecedents']=='Banatan Cream 50GM'])
                r1_5=r1_5.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_5['Recommendations'].unique())
                dfr=(r1_5['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Canthex 10 Capsules':
                r1_6=(rules.loc[rules['antecedents']=='Canthex 10 Capsules'])
                r1_6=r1_6.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_6['Recommendations'].unique())
                dfr=(r1_6['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Es Body Wash 200ML':
                r1_7=(rules.loc[rules['antecedents']=='Es Body Wash 200ML'])
                r1_7=r1_7.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_7['Recommendations'].unique())
                dfr=(r1_7['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Melawash 100 ML':
                r1_8=(rules.loc[rules['antecedents']=='Melawash 100 ML'])
                r1_8=r1_8.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_8['Recommendations'].unique())
                dfr=(r1_8['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Triobloc Cream 25GM':
                r1_9=(rules.loc[rules['antecedents']=='Triobloc Cream 25GM'])
                r1_9=r1_9.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_9['Recommendations'].unique())
                dfr=(r1_9['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Zinikam Cream 200 G':
                r1_10=(rules.loc[rules['antecedents']=='Zinikam Cream 200 G'])
                r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_10['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=="""Nixiyax 15 'S""":
                r1_11=(rules.loc[rules['antecedents']=="""Nixiyax 15 'S"""])
                r1_11=r1_11.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_11['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            
            
                
                
                
        
        if option=='AHMEDABAD':
            d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
            d['PRODUCTS'] = d['PRODUCTS'].str.strip()
            basket = (d[d['DEPOT'] =='AHMEDABAD']
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            #st.write(frequent_itemsets.head())
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(rules.head())
            c1=rules['antecedents'].unique()
            cl=list(c1)
            sb=st.selectbox('select product',cl)
            if sb=='Acnemoist Cream 60GM':
                r1=(rules.loc[rules['antecedents']=='Acnemoist Cream 60GM'])
                r1=r1.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1['Recommendations'].unique())
                dfr=(r1['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Aknay Bar 100GM':
                r1_2=(rules.loc[rules['antecedents']=='Aknay Bar 100GM'])
                r1_2=r1_2.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_2['Recommendations'].unique())
                dfr=(r1_2['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Solution 100Ml':
                r1_3=(rules.loc[rules['antecedents']=='Androanagen Solution 100Ml'])
                r1_3=r1_3.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_3['Recommendations'].unique())
                dfr=(r1_3['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Tablets 10S':
                r1_4=(rules.loc[rules['antecedents']=='Androanagen Tablets 10S'])
                r1_4=r1_4.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_4['Recommendations'].unique())
                dfr=(r1_4['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Banatan Cream 50GM':
                r1_5=(rules.loc[rules['antecedents']=='Banatan Cream 50GM'])
                r1_5=r1_5.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_5['Recommendations'].unique())
                dfr=(r1_5['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Canthex 10 Capsules':
                r1_6=(rules.loc[rules['antecedents']=='Canthex 10 Capsules'])
                r1_6=r1_6.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_6['Recommendations'].unique())
                dfr=(r1_6['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Es Body Wash 200ML':
                r1_7=(rules.loc[rules['antecedents']=='Es Body Wash 200ML'])
                r1_7=r1_7.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_7['Recommendations'].unique())
                dfr=(r1_7['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Melawash 100 ML':
                r1_8=(rules.loc[rules['antecedents']=='Melawash 100 ML'])
                r1_8=r1_8.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_8['Recommendations'].unique())
                dfr=(r1_8['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Triobloc Cream 25GM':
                r1_9=(rules.loc[rules['antecedents']=='Triobloc Cream 25GM'])
                r1_9=r1_9.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_9['Recommendations'].unique())
                dfr=(r1_9['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Zinikam Cream 200 G':
                r1_10=(rules.loc[rules['antecedents']=='Zinikam Cream 200 G'])
                r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_10['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\ Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=="""Nixiyax 15 'Sn""":
                r1_11=(rules.loc[rules['antecedents']=="""Nixiyax 15 'S"""])
                r1_11=r1_11.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_11['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
        
        if option=='AMBALA':
            d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
            d['PRODUCTS'] = d['PRODUCTS'].str.strip()
            basket = (d[d['DEPOT'] =='AMBALA']
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            #st.write(frequent_itemsets.head())
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(rules.head())
            c1=rules['antecedents'].unique()
            cl=list(c1)
            sb=st.selectbox('select product',cl)
            if sb=='Acnemoist Cream 60GM':
                r1=(rules.loc[rules['antecedents']=='Acnemoist Cream 60GM'])
                r1=r1.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1['Recommendations'].unique())
                dfr=(r1['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Aknay Bar 100GM':
                r1_2=(rules.loc[rules['antecedents']=='Aknay Bar 100GM'])
                r1_2=r1_2.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_2['Recommendations'].unique())
                dfr=(r1_2['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Solution 100Ml':
                r1_3=(rules.loc[rules['antecedents']=='Androanagen Solution 100Ml'])
                r1_3=r1_3.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_3['Recommendations'].unique())
                dfr=(r1_3['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Tablets 10S':
                r1_4=(rules.loc[rules['antecedents']=='Androanagen Tablets 10S'])
                r1_4=r1_4.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_4['Recommendations'].unique())
                dfr=(r1_4['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Banatan Cream 50GM':
                r1_5=(rules.loc[rules['antecedents']=='Banatan Cream 50GM'])
                r1_5=r1_5.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_5['Recommendations'].unique())
                dfr=(r1_5['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Canthex 10 Capsules':
                r1_6=(rules.loc[rules['antecedents']=='Canthex 10 Capsules'])
                r1_6=r1_6.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_6['Recommendations'].unique())
                dfr=(r1_6['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Es Body Wash 200ML':
                r1_7=(rules.loc[rules['antecedents']=='Es Body Wash 200ML'])
                r1_7=r1_7.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_7['Recommendations'].unique())
                dfr=(r1_7['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Melawash 100 ML':
                r1_8=(rules.loc[rules['antecedents']=='Melawash 100 ML'])
                r1_8=r1_8.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_8['Recommendations'].unique())
                dfr=(r1_8['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Triobloc Cream 25GM':
                r1_9=(rules.loc[rules['antecedents']=='Triobloc Cream 25GM'])
                r1_9=r1_9.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_9['Recommendations'].unique())
                dfr=(r1_9['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Zinikam Cream 200 G':
                r1_10=(rules.loc[rules['antecedents']=='Zinikam Cream 200 G'])
                r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_10['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=="""Nixiyax 15 'S""":
                r1_11=(rules.loc[rules['antecedents']=="""Nixiyax 15 'S"""])
                r1_11=r1_11.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_11['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
        

        if option=='GHAZIABAD':
            d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
            d['PRODUCTS'] = d['PRODUCTS'].str.strip()
            basket = (d[d['DEPOT'] =='GHAZIABAD']
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            #st.write(frequent_itemsets.head())
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(rules.head())
            c1=rules['antecedents'].unique()
            cl=list(c1)
            sb=st.selectbox('select product',cl)
            if sb=='Acnemoist Cream 60GM':
                r1=(rules.loc[rules['antecedents']=='Acnemoist Cream 60GM'])
                r1=r1.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1['Recommendations'].unique())
                dfr=(r1['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Aknay Bar 100GM':
                r1_2=(rules.loc[rules['antecedents']=='Aknay Bar 100GM'])
                r1_2=r1_2.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_2['Recommendations'].unique())
                dfr=(r1_2['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Solution 100Ml':
                r1_3=(rules.loc[rules['antecedents']=='Androanagen Solution 100Ml'])
                r1_3=r1_3.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_3['Recommendations'].unique())
                dfr=(r1_3['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Tablets 10S':
                r1_4=(rules.loc[rules['antecedents']=='Androanagen Tablets 10S'])
                r1_4=r1_4.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_4['Recommendations'].unique())
                dfr=(r1_4['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Banatan Cream 50GM':
                r1_5=(rules.loc[rules['antecedents']=='Banatan Cream 50GM'])
                r1_5=r1_5.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_5['Recommendations'].unique())
                dfr=(r1_5['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Canthex 10 Capsules':
                r1_6=(rules.loc[rules['antecedents']=='Canthex 10 Capsules'])
                r1_6=r1_6.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_6['Recommendations'].unique())
                dfr=(r1_6['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Es Body Wash 200ML':
                r1_7=(rules.loc[rules['antecedents']=='Es Body Wash 200ML'])
                r1_7=r1_7.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_7['Recommendations'].unique())
                dfr=(r1_7['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Melawash 100 ML':
                r1_8=(rules.loc[rules['antecedents']=='Melawash 100 ML'])
                r1_8=r1_8.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_8['Recommendations'].unique())
                dfr=(r1_8['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Triobloc Cream 25GM':
                r1_9=(rules.loc[rules['antecedents']=='Triobloc Cream 25GM'])
                r1_9=r1_9.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_9['Recommendations'].unique())
                dfr=(r1_9['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Zinikam Cream 200 G':
                r1_10=(rules.loc[rules['antecedents']=='Zinikam Cream 200 G'])
                r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_10['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=="""Nixiyax 15 'S""":
                r1_11=(rules.loc[rules['antecedents']=="""Nixiyax 15 'S"""])
                r1_11=r1_11.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_11['Recommendations'].unique())
                dfr=(r1_11['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))


        if option=='KOLKATTA':
            d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
            d['PRODUCTS'] = d['PRODUCTS'].str.strip()
            basket = (d[d['DEPOT'] =='KOLKATTA']
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            #st.write(frequent_itemsets.head())
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(rules.head())
            c1=rules['antecedents'].unique()
            cl=list(c1)
            sb=st.selectbox('select product',cl)
            if sb=='Acnemoist Cream 60GM':
                r1=(rules.loc[rules['antecedents']=='Acnemoist Cream 60GM'])
                r1=r1.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1['Recommendations'].unique())
                dfr=(r1['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Aknay Bar 100GM':
                r1_2=(rules.loc[rules['antecedents']=='Aknay Bar 100GM'])
                r1_2=r1_2.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_2['Recommendations'].unique())
                dfr=(r1_2['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Solution 100Ml':
                r1_3=(rules.loc[rules['antecedents']=='Androanagen Solution 100Ml'])
                r1_3=r1_3.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_3['Recommendations'].unique())
                dfr=(r1_3['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Tablets 10S':
                r1_4=(rules.loc[rules['antecedents']=='Androanagen Tablets 10S'])
                r1_4=r1_4.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_4['Recommendations'].unique())
                dfr=(r1_4['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Banatan Cream 50GM':
                r1_5=(rules.loc[rules['antecedents']=='Banatan Cream 50GM'])
                r1_5=r1_5.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_5['Recommendations'].unique())
                dfr=(r1_5['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Canthex 10 Capsules':
                r1_6=(rules.loc[rules['antecedents']=='Canthex 10 Capsules'])
                r1_6=r1_6.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_6['Recommendations'].unique())
                dfr=(r1_6['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Es Body Wash 200ML':
                r1_7=(rules.loc[rules['antecedents']=='Es Body Wash 200ML'])
                r1_7=r1_7.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_7['Recommendations'].unique())
                dfr=(r1_7['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Melawash 100 ML':
                r1_8=(rules.loc[rules['antecedents']=='Melawash 100 ML'])
                r1_8=r1_8.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_8['Recommendations'].unique())
                dfr=(r1_8['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Triobloc Cream 25GM':
                r1_9=(rules.loc[rules['antecedents']=='Triobloc Cream 25GM'])
                r1_9=r1_9.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_9['Recommendations'].unique())
                dfr=(r1_9['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Zinikam Cream 200 G':
                r1_10=(rules.loc[rules['antecedents']=='Zinikam Cream 200 G'])
                r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_10['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=="""Nixiyax 15 'S""":
                r1_11=(rules.loc[rules['antecedents']=="""Nixiyax 15 'S"""])
                r1_11=r1_11.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_11['Recommendations'].unique())
                dfr=(r1_11['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
        

        if option=='PUNE':
            d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
            d['PRODUCTS'] = d['PRODUCTS'].str.strip()
            basket = (d[d['DEPOT'] =='PUNE']
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            #st.write(frequent_itemsets.head())
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(rules.head())
            c1=rules['antecedents'].unique()
            cl=list(c1)
            sb=st.selectbox('select product',cl)
            if sb=='Acnemoist Cream 60GM':
                r1=(rules.loc[rules['antecedents']=='Acnemoist Cream 60GM'])
                r1=r1.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1['Recommendations'].unique())
                dfr=(r1['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Aknay Bar 100GM':
                r1_2=(rules.loc[rules['antecedents']=='Aknay Bar 100GM'])
                r1_2=r1_2.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_2['Recommendations'].unique())
                dfr=(r1_2['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Solution 100Ml':
                r1_3=(rules.loc[rules['antecedents']=='Androanagen Solution 100Ml'])
                r1_3=r1_3.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_3['Recommendations'].unique())
                dfr=(r1_3['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Tablets 10S':
                r1_4=(rules.loc[rules['antecedents']=='Androanagen Tablets 10S'])
                r1_4=r1_4.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_4['Recommendations'].unique())
                dfr=(r1_4['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Banatan Cream 50GM':
                r1_5=(rules.loc[rules['antecedents']=='Banatan Cream 50GM'])
                r1_5=r1_5.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_5['Recommendations'].unique())
                dfr=(r1_5['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Canthex 10 Capsules':
                r1_6=(rules.loc[rules['antecedents']=='Canthex 10 Capsules'])
                r1_6=r1_6.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_6['Recommendations'].unique())
                dfr=(r1_6['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Es Body Wash 200ML':
                r1_7=(rules.loc[rules['antecedents']=='Es Body Wash 200ML'])
                r1_7=r1_7.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_7['Recommendations'].unique())
                dfr=(r1_7['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Melawash 100 ML':
                r1_8=(rules.loc[rules['antecedents']=='Melawash 100 ML'])
                r1_8=r1_8.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_8['Recommendations'].unique())
                dfr=(r1_8['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Triobloc Cream 25GM':
                r1_9=(rules.loc[rules['antecedents']=='Triobloc Cream 25GM'])
                r1_9=r1_9.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_9['Recommendations'].unique())
                dfr=(r1_9['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Zinikam Cream 200 G':
                r1_10=(rules.loc[rules['antecedents']=='Zinikam Cream 200 G'])
                r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_10['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=="""Nixiyax 15 'S""":
                r1_11=(rules.loc[rules['antecedents']=="""Nixiyax 15 'S"""])
                r1_11=r1_11.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_11['Recommendations'].unique())
                dfr=(r1_11['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
        

        if option=='GUWAHATI':
            d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
            d['PRODUCTS'] = d['PRODUCTS'].str.strip()
            basket = (d[d['DEPOT'] =='GUWAHATI']
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            #frequent_itemsets['itemsets']=frequent_itemsets['itemsets'].astype(str)
            #frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].map(lambda x: x.lstrip('frozenset'))
            #st.write(frequent_itemsets.head())
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(rules.head())
            c1=rules['antecedents'].unique()
            cl=list(c1)
            sb=st.selectbox('select product',cl)
            if sb=='Acnemoist Cream 60GM':
                r1=(rules.loc[rules['antecedents']=='Acnemoist Cream 60GM'])
                r1=r1.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1['Recommendations'].unique())
                dfr=(r1['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Aknay Bar 100GM':
                r1_2=(rules.loc[rules['antecedents']=='Aknay Bar 100GM'])
                r1_2=r1_2.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_2['Recommendations'].unique())
                dfr=(r1_2['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Solution 100Ml':
                r1_3=(rules.loc[rules['antecedents']=='Androanagen Solution 100Ml'])
                r1_3=r1_3.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_3['Recommendations'].unique())
                dfr=(r1_3['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Tablets 10S':
                r1_4=(rules.loc[rules['antecedents']=='Androanagen Tablets 10S'])
                r1_4=r1_4.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_4['Recommendations'].unique())
                dfr=(r1_4['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Banatan Cream 50GM':
                r1_5=(rules.loc[rules['antecedents']=='Banatan Cream 50GM'])
                r1_5=r1_5.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_5['Recommendations'].unique())
                dfr=(r1_5['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Canthex 10 Capsules':
                r1_6=(rules.loc[rules['antecedents']=='Canthex 10 Capsules'])
                r1_6=r1_6.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_6['Recommendations'].unique())
                dfr=(r1_6['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Es Body Wash 200ML':
                r1_7=(rules.loc[rules['antecedents']=='Es Body Wash 200ML'])
                r1_7=r1_7.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_7['Recommendations'].unique())
                dfr=(r1_7['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Melawash 100 ML':
                r1_8=(rules.loc[rules['antecedents']=='Melawash 100 ML'])
                r1_8=r1_8.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_8['Recommendations'].unique())
                dfr=(r1_8['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Triobloc Cream 25GM':
                r1_9=(rules.loc[rules['antecedents']=='Triobloc Cream 25GM'])
                r1_9=r1_9.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_9['Recommendations'].unique())
                dfr=(r1_9['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Zinikam Cream 200 G':
                r1_10=(rules.loc[rules['antecedents']=='Zinikam Cream 200 G'])
                r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_10['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=="""Nixiyax 15 'S""":
                r1_11=(rules.loc[rules['antecedents']=="""Nixiyax 15 'S"""])
                r1_11=r1_11.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_11['Recommendations'].unique())
                dfr=(r1_11['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
        

        if option=='ERNAKULAM':
            d['STOCKIEST NAME'] = d['STOCKIEST NAME'].str.strip()
            d['PRODUCTS'] = d['PRODUCTS'].str.strip()
            basket = (d[d['DEPOT'] =='ERNAKULAM']
                    .groupby(['Unique Id', 'PRODUCTS'])['RECEIPT']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Unique Id'))
            def encode_units(x):
                if int(x) <= int(0):
                    return 0
                if int(x) >= int(1):
                    return 1
            basket_sets = basket.applymap(encode_units)
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            #frequent_itemsets['itemsets']=frequent_itemsets['itemsets'].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(frequent_itemsets.head())
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
            #st.write(rules.head())
            c1=rules['antecedents'].unique()
            cl=list(c1)
            sb=st.selectbox('select product',cl)
            if sb=='Acnemoist Cream 60GM':
                r1=(rules.loc[rules['antecedents']=='Acnemoist Cream 60GM'])
                r1=r1.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1['Recommendations'].unique())
                dfr=(r1['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Aknay Bar 100GM':
                r1_2=(rules.loc[rules['antecedents']=='Aknay Bar 100GM'])
                r1_2=r1_2.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_2['Recommendations'].unique())
                dfr=(r1_2['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Solution 100Ml':
                r1_3=(rules.loc[rules['antecedents']=='Androanagen Solution 100Ml'])
                r1_3=r1_3.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_3['Recommendations'].unique())
                dfr=(r1_3['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Androanagen Tablets 10S':
                r1_4=(rules.loc[rules['antecedents']=='Androanagen Tablets 10S'])
                r1_4=r1_4.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_4['Recommendations'].unique())
                dfr=(r1_4['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Banatan Cream 50GM':
                r1_5=(rules.loc[rules['antecedents']=='Banatan Cream 50GM'])
                r1_5=r1_5.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_5['Recommendations'].unique())
                dfr=(r1_5['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Canthex 10 Capsules':
                r1_6=(rules.loc[rules['antecedents']=='Canthex 10 Capsules'])
                r1_6=r1_6.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_6['Recommendations'].unique())
                dfr=(r1_6['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Es Body Wash 200ML':
                r1_7=(rules.loc[rules['antecedents']=='Es Body Wash 200ML'])
                r1_7=r1_7.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_7['Recommendations'].unique())
                dfr=(r1_7['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Melawash 100 ML':
                r1_8=(rules.loc[rules['antecedents']=='Melawash 100 ML'])
                r1_8=r1_8.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_8['Recommendations'].unique())
                dfr=(r1_8['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Triobloc Cream 25GM':
                r1_9=(rules.loc[rules['antecedents']=='Triobloc Cream 25GM'])
                r1_9=r1_9.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_9['Recommendations'].unique())
                dfr=(r1_9['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=='Zinikam Cream 200 G':
                r1_10=(rules.loc[rules['antecedents']=='Zinikam Cream 200 G'])
                r1_10=r1_10.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_10['Recommendations'].unique())
                dfr=(r1_10['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))
            if sb=="""Nixiyax 15 'S""":
                r1_11=(rules.loc[rules['antecedents']=="""Nixiyax 15 'S"""])
                r1_11=r1_11.rename(columns={'consequents':'Recommendations'},inplace=False)
                #st.write(r1_11['Recommendations'].unique())
                dfr=(r1_11['Recommendations'].unique())
                k=pd.DataFrame(dfr)
                k.columns= ['Recommendations']
                st.write(k['Recommendations'].unique())
                st.title('Verify Your Plan')
                data3=st.file_uploader('Upload The Dataset',type=['csv'],key='1')
                if data3 is not None:
                    dx=pd.read_csv(data3)
                    st.success('Data Successfully Uploaded')
                    st.write(dx)
                    st.write("\n Recommend you to add this product in ur list\n",pd.concat([k,dx]).drop_duplicates(keep=False))


        

        st.title('Monthly Sales Analysis')
    #if st.button('click'):
        depot=d['DEPOT'].unique()
        x1=st.selectbox('Select the Depot:',list(depot))
        #s1=(d[d['DEPOT'] == x1])


        if x1=='DELHI':
            gk = d.groupby('DEPOT')
            a=gk.get_group('DELHI')
            s=(a['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2=st.selectbox('select the stockiest name:',list(s))
        elif x1=='AHMEDABAD':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('AHMEDABAD')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_1=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='ERNAKULAM':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('ERNAKULAM')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_2=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='GUWAHATI':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('GUWAHATI')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_3=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='AMBALA':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('AMBALA')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_4=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='PUNE':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('PUNE')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_5=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='KOLKATTA':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('KOLKATTA')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_6=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='GHAZIABAD':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('GHAZIABAD')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_7=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='HYDERABAD':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('HYDERABAD')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_8=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='VIJAYAWADA':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('VIJAYAWADA')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_9=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='BANGALORE':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('BANGALORE')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_10=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='JAIPUR':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('JAIPUR')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_11=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='HUBLI':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('HUBLI')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_11=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='MUMBAI':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('MUMBAI')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_12=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='BHOPAL':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('BHOPAL')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_13=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='RAIPUR':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('RAIPUR')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_14=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='RANCHI':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('RANCHI')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_15=st.selectbox('select the stockiest name:',list(s1))
        elif x1=='CHENNAI':
            gk1 = d.groupby('DEPOT')
            a1=gk1.get_group('CHENNAI')
            s1=(a1['STOCKIEST NAME'].unique())
            #stockiest=d['STOCKIEST NAME'].unique()
            x2_16=st.selectbox('select the stockiest name:',list(s1))
            



        #stockiest=d['STOCKIEST NAME'].unique()
        #x2=st.selectbox('select the stockiest name:',list(stockiest))

        products=d['PRODUCTS'].unique()
        x3=st.selectbox('select the Product:',list(products))

        #month=d['MONTH'].unique()
        #x4=st.selectbox('select the month:',list(month))
        #st.write(x4)

        #filter_data = st.text_input('Enter the Branch Name:')
        #filtered_depot=(d[d['DEPOT'] == x1])
        if x1=='AHMEDABAD': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_1) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='DELHI': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a["STOCKIEST NAME"]==x2) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='MUMBAI': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_12) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='CHENNAI': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_16) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='RANCHI': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_15) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='RAIPUR': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_14) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='BHOPAL': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_13) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='HUBLI': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_11) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='JAIPUR': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_10) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='BANGALORE': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_8) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='VIJAYAWADA': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_9) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='HYDERABAD': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_8) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='GHAZIABAD': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_7) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='AMBALA': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_4) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='KOLKATTA': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_6) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='PUNE': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_5) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif x1=='GUWAHATI': 
            filtered_depot = d[(d["DEPOT"]== x1) & (a1["STOCKIEST NAME"]==x2_3) & (d['PRODUCTS']==x3)] #& (d['MONTH']==x4)]
            st.write("Customers filtered by ",x1)
            st.write(filtered_depot)
            df=filtered_depot
            df1=pd.DataFrame(df)
            st.write ('monthly sales of:',x3)
            df1.plot(x='MONTH', y=['RECEIPT','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.title('Monthly Sales Revenue')
            df1.plot(x='MONTH', y=['SALES VALUE','PRODUCTS'], kind="bar") 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()


        #st.title('Look Alike Analysis')
        #lk=d
        #st.write(lk.head())
        #lk.drop('MONTH',axis=1,inplace=True)
        #lk.drop('FY',axis=1,inplace=True)
        #data_types_dict=dict(lk.dtypes)
        #label_encoder_collection= {}
        #for col_name, data_type in data_types_dict.items():
            #if data_type=='object':
                #le=LabelEncoder()
                #lk[col_name]=le.fit_transform(lk[col_name])
        #sc=MinMaxScaler()
        #lk['RECEIPT VALUE'] =  sc.fit_transform(lk[['RECEIPT VALUE']])
        #lk['PTS'] = sc.fit_transform(lk[['PTS']])
        #lk['SALES VALUE'] = sc.fit_transform(lk[['SALES VALUE']])
        #lk['CLOSING VALUE'] = sc.fit_transform(lk[['CLOSING VALUE']])
        #lk['TOTAL CLOSING VALUE'] = sc.fit_transform(lk[['TOTAL CLOSING VALUE']])
        #lk['0PENING VALUE']=pd.to_numeric(lk['0PENING VALUE'],errors='coerce')
        #lk['0PENING VALUE'] = sc.fit_transform(lk[['0PENING VALUE']])
        #st.write(lk.head())

        #nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='cosine').fit(lk)
        #distances, indices = nbrs.kneighbors(lk)
        #dist_df = pd.DataFrame(distances)
        #dist_df.reset_index(inplace = True, drop = True)
                #st.write(dist_df.head())
        #dist = dist_df * 1000
                # dist_ = pd.to_numeric(dist)
                # dist_= dist.round(decimals=2)


            
                #st.write(dist)
                
        #ind_df=pd.DataFrame(indices)
        #ind_df.reset_index(inplace = True, drop = True)
                #st.write(test_lk)
                #st.write(ind_df)
        
        #lk_output_ind=pd.concat([d,ind_df], axis=1)


        #lk_output=pd.concat([d,ind_df], axis=1)
        #lk_output.reset_index(inplace = True, drop = True)
        #st.write('Customer Information with Similarity Scores: ', lk_output)



        #Customer_id1 = st.text_input('UID:',step=1)
        #user_value=Customer_id1
                # user_value=Customer_id1
        #temp3=pd.DataFrame()
        #for i in range(0,11):
            #temp=lk_output_ind[[0,1,2,3,4,5,6,7,8,9,10]].iloc[user_value,i]
            #temp2=pd.DataFrame(lk_output_ind.loc[lk_output_ind.index==temp])
            #temp2.drop('Id', inplace=True, axis=1)
            #temp3=temp3.append(temp2)
                    #st.write(temp2)
            #st.write('Top 10 Similar Customers:',temp3)




        

        



        


        
        
if __name__ == '__main__':
    main()

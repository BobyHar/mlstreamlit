import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormapi
import datetime

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Machine Learning K-Means')

def app():
    st.subheader('Input dataset dengan tipe xls')
    uploaded_file = st.file_uploader('Choose a dataset',type='xlsx')

    #df=pd.read_csv("marketing_campaign.csv")
    if uploaded_file:
        st.markdown('___')
        df = pd.read_excel(uploaded_file, engine='openpyxl')



        st.subheader('Data Frame')
        st.dataframe(df)

        
        st.write("""MODELING""")

        # Menentukan jumlah kluster dengan metode elbow
        inertia = []
        for n in range(1 , 11):
            algorithm = KMeans(n_clusters = n, init='k-means++', n_init = 10 , max_iter=300, random_state= 111)
            algorithm.fit(scaled_ds)
            inertia.append(algorithm.inertia_)
        # Plot elbow
        plt.figure(1 , figsize = (15 ,6))
        plt.plot(np.arange(1 , 11) , inertia , 'o')
        plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
        plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
        plt.show()


        #Initiating the KMeans model 
        fig, ax = plt.subplots()
        km = KMeans(n_clusters = 4, init='k-means++', n_init = 10 , max_iter=300, random_state= 111)
        # fit model and predict clusters
        fitModel = km.fit(scaled_ds)
        predicted = km.predict(scaled_ds)
        #Adding the Clusters feature to the orignal dataframe.
        scaled_ds["Cluster"]= predicted
        df["Cluster"]= predicted
        centroids = km.cluster_centers_

        #Plotting countplot of clusters
        pal = ["#FD8A8A","#F1F7B5", "#A8D1D1","#9EA1D4"]
        pl = sns.countplot(x=scaled_ds["Cluster"], palette= pal)
        


        fig, ax = plt.subplots()
        pl = sns.scatterplot(data = scaled_ds, x = scaled_ds["TotalSpending"], y = scaled_ds["Income"], hue = scaled_ds["Cluster"], palette= pal)
        plt.scatter(
            centroids[:, 17],
            centroids[:, 2],
            s=200,
            linewidths=3,
            color="yellow",
            zorder=10,
        )
        pl.set_title("Total spending vs total income")
        plt.legend()
        plt.show()
        st.pyplot(fig)

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'TotalSpending'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntWines'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntFruits'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntMeatProducts'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntFishProducts'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntGoldProds'].sum())
        arrayList

        
        df.info()

        info = ['Education','Marital_Status', 'NumDealsPurchases', 'NumWebPurchases', 'NumWebVisitsMonth', 'NumCatalogPurchases','NumStorePurchases', 'Children', 'Parent', 'Age']
        for i in info:
            plt.figure()
            sns.jointplot(x=df[i], y=df["TotalSpending"], hue =df["Cluster"], kind="kde", palette=pal)
            st.pyplot(plt.show())


        plt.figure()
        sns.jointplot(x=df['Parent'], y=df["TotalSpending"], hue =df["Cluster"], kind="kde", palette=pal)
        st.pyplot(plt.show())
     

if __name__=='__main__':
    app()


# %%
#数据清洗和热力图
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# 导入movie信息
Path="C:/Users/vv/Desktop/movie"
movie = pd.read_csv(os.path.join(Path,'tmdb_5000_movies.csv'))

#创建movie的str类型数据列表
movie_list=[]
for colname,colvalue in movie.iteritems():
    if type(colvalue[1])==str:
        movie_list.append(colname)
#提取movie表数据不是str的属性index
num_list=movie.columns.difference(movie_list)
#导入数值
movie_num=movie[num_list]
movie_num.head()
#补缺失值
movie_num=movie_num.fillna(value=0,axis=1)
movie_num.isna().sum()

# feature的热力图，使用皮尔森相关系数
f, ax = plt.subplots(figsize=(12,10))
plt.title('Pearson Correlation of Movie Features')
sns.heatmap(movie_num.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True,
           cmap="YlGnBu", linecolor='black', annot=True)

X=movie_num.values
# movie_num.to_csv("C:/Users/vv/Desktop/movie_X.csv")

#%%
# #PCA与Kmean
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 导入movie信息
Path="C:/Users/vv/Desktop/movie"
movie = pd.read_csv(os.path.join(Path,'tmdb_5000_movies.csv'))

#创建movie的str类型数据列表
movie_list=[]
for colname,colvalue in movie.iteritems():
    if type(colvalue[1])==str:
        movie_list.append(colname)
#提取movie表数据不是str的属性index
num_list=movie.columns.difference(movie_list)
#导入数值
movie_num=movie[num_list]
movie_num.head()
#补缺失值
movie_num=movie_num.fillna(value=0,axis=1)
movie_num.isna().sum()

X=movie_num.values
# 导出non-string数据
# movie_num.to_csv("C:/Users/vv/Desktop/movie_X.csv")

# 展示Raw data
# plt.figure(figsize=(10,8))
# plt.scatter(movie_num.iloc[:,2],movie_num.iloc[:,3])
# plt.xlabel('Popularity')
# plt.ylabel('revenue')
# plt.title('Raw data')
# plt.show()

#数据标准化
X_std=StandardScaler().fit_transform(X)

#根据手肘图来选择component
ppca=PCA()
ppca.fit(X_std)
plt.plot(np.cumsum(ppca.explained_variance_ratio_))
plt.title("Explained Variance by components")
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

#使用pca降维 
pca = PCA(n_components=3)
#用训练器数据拟合分类器模型
pca.fit(X_std) 

# 查看每个feature的explained variance
# pca.explained_variance_ratio_


#输出每一列属性所占特征方差的百分比，和总和
# print(pca.explained_variance_ratio_)
# c=pca.explained_variance_ratio_.sum()
# print(c)

# 计算数据集中每个元素对应component的分数
pca.transform(X_std)
scores_pca=pca.transform(X_std)

#通过手肘图确定K-means的簇数k
wcss=[]
for i in range(1,21):
    kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10,8))
plt.plot(range(1,21),wcss,marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means with PCA Clustering')
plt.show()

#k =3 or 4

#K-means 与 PCA的结合——使用主成分分数拟合K-mean模型
kmeans_pca= KMeans(n_clusters=4,init='k-means++',random_state=42)
kmeans_pca.fit(scores_pca)

#获得clustering结果
pca_cluster=kmeans_pca.fit_predict(scores_pca)

#构建包括cluster结果的数据框
df_pca_kmeans=pd.concat([movie_num.reset_index(drop =True),pd.DataFrame(scores_pca)],axis=1)
df_pca_kmeans.columns.values[-3:]=['Component1','Component2','Component3']
df_pca_kmeans['Segment K-means PCA']=kmeans_pca.labels_

#统计每个簇的数量和占比
pca_cluster_count=pd.DataFrame(df_pca_kmeans['id'].groupby(
    df_pca_kmeans['Segment K-means PCA']).count()).T.rename({"id":"count"})

pca_cluster_ratio=(pca_cluster_count/len(df_pca_kmeans)).round(4).rename({"count":"percent"})

#可视化每个簇的数量
pca_cluster_count.plot(kind='bar',title='The count of each cluster',figsize=(8,4))
# pca_cluster_ratio.plot(kind='pie',title='The percentage of each cluster',figsize=(8,5),subplots=True)

# 创建 cluster新列，并直接映射四个集群
df_pca_kmeans['Segment']=df_pca_kmeans['Segment K-means PCA'].map({0:'first',1:'second',2:'third',3:'fourth'})

#选择两个component进行clustering可视化
#我们可以确认前两个分量会比第三个分量解释更多的方差
x_axis=df_pca_kmeans['Component1']
y_axis=df_pca_kmeans['Component2']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis,y_axis, hue=df_pca_kmeans['Segment'],palette=['g','r','c','m'])
plt.title('Clusters by PCA Components')
plt.show()
# %%

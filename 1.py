# %%
from cProfile import label
from typing_extensions import runtime
from matplotlib.lines import _LineStyle
import scipy
import pickle
import os
import json
from scipy.spatial.distance import squareform, pdist
import scipy.cluster.hierarchy as shc
import numpy as np
from functools import partial
from platform import node
from itsdangerous import json
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from path import Path
import sklearn
from sklearn import cluster
from sympy import mobius
email_network = nx.read_edgelist(
    "C:\\Users\\vv\\Desktop\\email-Eu-core.txt", create_using=nx.DiGraph())

largest_com = [email_network.subgraph(
    c).copy() for c in nx.weakly_connected_components(email_network)]
pr_dict = nx.pagerank(email_network)

pr_df = pd.DataFrame.from_dict(pr_dict, orient="index")
pr_df.columns = ["pr_value"]
pr_df.sort_values(by="pr_value").head(20)
pr_df.head(20)


def get_nodesize_pagerank(pagerank, min_size, max_size):
    nodesize_list = []
    pr_max = max(pagerank.value())
    for node, pr in pagerank.item():
        nodesize = max(max_size - min_size)*pr/pr_max+min_size
        nodesize_list.append(nodesize)
    return nodesize_list


fig, ax = plt.subplots(figsize=(24, 16))
pos_emails3 = nx.kamada_kawai_layout(email_network)
ax.axis("off")
plt.box(False)
nx.draw(email_network, node_size=get_nodesize_pagerank(pr_dict, 1, 100),
        node_color="green", edge_color="#D8D8D8", width=.3, pos=pos_emails3, ax=ax)


# %%
#数据清洗
import numpy as np
from functools import partial
from platform import node
from itsdangerous import json
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from path import Path
Path = "C:/Users/vv/Desktop/movie"
print(os.listdir(Path))
credit = pd.read_csv(os.path.join(Path, 'tmdb_5000_credits.csv'))
movie = pd.read_csv(os.path.join(Path, 'tmdb_5000_movies.csv'))


credit.columns = ['id', 'tittle', 'cast', 'crew']
movie_df = movie.merge(credit, on='id')
del movie
del credit

movie_df["release_date"] = pd.to_datetime(movie_df['release_date'])
movie_df['release_year'] = movie_df['release_date'].dt.year
movie_df['release_month'] = movie_df['release_date'].dt.month_name()
del movie_df['release_date']

json_columns = {'cast', 'crew', 'genres', 'keywords',
                'production_countries', 'production_companies', 'spoken_languages'}

for c in json_columns:
    movie_df[c] = movie_df[c].apply(json.loads)
    if c != "crew":
        movie_df[c] = movie_df[c].apply(lambda row: [x["name"] for x in row])


def get_job(job, row):
    person_name = [x['name'] for x in row if x['job'] == job]
    return person_name[0] if len(person_name)else np.nan


movie_df["director"] = movie_df["crew"].apply(partial(get_job, "Director"))
movie_df["writer"] = movie_df["crew"].apply(partial(get_job, "Writer"))
movie_df["producer"] = movie_df["crew"].apply(partial(get_job, "Producer"))
del movie_df["crew"]

movie_df["profit"] = movie_df["revenue"] - movie_df["budget"]

for col in ["runtime", "release_year", "release_month"]:
    movie_df[col] = movie_df[col].fillna(movie_df[col].mode().iloc[0])

movie_df.to_csv("C:/Users/vv/Desktop/movie_df.csv")
# %%
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
print(os.listdir(Path))
credit = pd.read_csv(os.path.join(Path,'tmdb_5000_credits.csv'))
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
# movie_num.to_csv("C:/Users/vv/Desktop/movie_X.csv")

#计算每个属性的std
X_std=StandardScaler().fit_transform(X)
# #画出随着runtime，vote_count变化的hexbin图
# movie.plot(x="runtime",y="vote_count",kind='hexbin',figsize=(12,8))

#使用pca降维
pca = PCA(n_components=4)
#用训练器数据拟合分类器模型
pca.fit(X_std) 

ppca=PCA()
ppca.fit(X_std)
plt.plot(np.cumsum(ppca.explained_variance_ratio_))
plt.title("Explained Variance by components")
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

#输出每一列属性所占特征方差的百分比，和总和
print(pca.explained_variance_ratio_)
c=pca.explained_variance_ratio_.sum()
print(c)


#PCA降维，数据存在x_9d
x_9d=pca.fit_transform(X_std)

#Kmean进行聚类，创建聚类对象，分3类
kmeans=KMeans(n_clusters=3)
#计算聚类中心并预测每个样本的聚类索引
X_clustered=kmeans.fit_predict(x_9d)
label_pre=kmeans.labels_

#画图
LABEL_COLOR = {0 : 'red',1 : 'green',2 : 'yellow'}
label_color = [LABEL_COLOR[l] for l in X_clustered]
#画图并显示出族中心
plt.figure(figsize=(7,7))
plt.scatter(x_9d[:,0],x_9d[:,2],c=label_color,cmap='viridis')
centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,2],c='black',s=200,alpha=0.5)
plt.show()

cluster_labels=pd.DataFrame(X_clustered,columns=['cluster'])
merge_data= pd.concat((movie_num,cluster_labels),axis=1)
merge_data.head()

cluster_count=pd.DataFrame(merge_data['id'].groupby(
    merge_data['cluster']).count()).T.rename({"id":"count"})

cluster_ratio=(cluster_count/len(merge_data)).round(4).rename({"count":"percent"})

cluster_features=[]
for line in range(3):
    label_data= merge_data[merge_data['cluster']== line]
    part1_data= label_data.iloc[:,1:8]
    part1_desc= part1_data.describe().round(3)
    merge_data_mean=part1_desc.iloc[2,:]

    cluster_features.append(merge_data_mean)  # 添加到列表中
    cluster_features

df=pd.DataFrame(x_9d)
df=df[[0,1,2]]
df['X_cluster']=X_clustered



# plt.figure(figsize=(10,8))
# plt.plot(range(1,6),pca.explained_variance_ratio_.cumsum(),marker='o',linestyle='- -')
# plt.title("Explained Variance by components")
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')

#将kmean的结果用seaborn投射 并显示聚类两两之间的关系
sns.pairplot(df, hue='X_cluster',palette='Dark2',diag_kind='hist',size=1.85)



# %%
##‘Popularity'’vote_average‘变量间kmean
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
print(os.listdir(Path))
credit = pd.read_csv(os.path.join(Path,'tmdb_5000_credits.csv'))
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

X=movie_num.iloc[:,[2,5]].values
print(X)

wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=42)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
Y=kmeans.fit_predict(X)

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label='C1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red',label='C2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='yellow',label='C3')
# plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='violet',label='C4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='Centroids')

plt.title('Movie Groups')
plt.xlabel('Popularity')
plt.ylabel('vote_average')
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import json


# 导入movie信息
Path="C:/Users/vv/Desktop/movie"
credit = pd.read_csv(os.path.join(Path,'tmdb_5000_credits.csv'))
movie = pd.read_csv(os.path.join(Path,'tmdb_5000_movies.csv'))


movie.drop(['homepage','tagline','id'],axis=1,inplace=True)
movie['runtime'].fillna(movie['runtime'].mean(),inplace=True)
movie.dropna(inplace=True)
movie.isnull().sum()
movie.duplicated().sum()

movie.describe()
# %%
#Popularity 与 revenue 的kmean
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

X=movie_num.iloc[:,[2,3]].values
print(X)

f, ax = plt.subplots(figsize=(12,10))
plt.title('Pearson Correlation of Movie Features')
sns.heatmap(movie_num.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True,
           cmap="YlGnBu", linecolor='black', annot=True)

kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
Y=kmeans.fit_predict(X)

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label='C1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red',label='C2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='yellow',label='C3')
# plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='violet',label='C4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='Centroids')

plt.title('Movie Groups')
plt.xlabel('Popularity')
plt.ylabel('Revenue')
plt.show()

# %%
#另一种PCA与Kmean
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

# 展示raw data
# plt.figure(figsize=(10,8))
# plt.scatter(movie_num.iloc[:,2],movie_num.iloc[:,3])
# plt.xlabel('Popularity')
# plt.ylabel('revenue')
# plt.title('Raw data')
# plt.show()
# plt.clf()

#计算每个属性的std
X_std=StandardScaler().fit_transform(X)

#根据手肘图来选择component
ppca=PCA()
ppca.fit(X_std)
plt.plot(np.cumsum(ppca.explained_variance_ratio_))
plt.title("Explained Variance by components")
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
plt.clf()

#使用pca降维 
pca = PCA(n_components=3)
#用训练器数据拟合分类器模型
pca.fit(X_std) 
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
plt.clf()
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

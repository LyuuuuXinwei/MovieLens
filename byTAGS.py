import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

tags = pd.read_csv('tags.csv', header=None,names=['movie_id', 'movie_tag'],usecols=[1,2])
movies = pd.read_csv('movies.csv', header=None,names=['movie_id', 'movie', 'genres'])
ratings = pd.read_csv('ratings.csv', header=None,names=['user_id', 'movie_id', 'rating'],usecols=[0,1,2])

item_tag_dict = {}
for movie in movies['movie_id'].tolist():
    item_tag_dict[movie] = {}

#得到电影-标签频数表
item_tags = tags.groupby('movie_id')
for item in item_tags:
    item_i_tags = item[1]['movie_tag'].value_counts().to_dict()
    item_tag_dict[item[0]] = item_i_tags
Q = pd.DataFrame.from_dict(item_tag_dict, orient='index')

#筛选
item_with_no_tags = [i for i in item_tag_dict.keys() if len(item_tag_dict[i]) == 0]
for item in item_with_no_tags:
    Q.loc[item] = np.nan
Q = Q.sort_index()


#得到电影-标签tf-idf相关度表
df = Q.count()
lnn = np.log(Q.shape[0])
idf = lnn - np.log(df)
Q = Q.fillna(0)
Q = Q*idf
Q = Q.div(np.sqrt(np.square(Q).sum(axis=1)), axis=0)
Q = Q.fillna(0)

#得到电影-人评分表
item_rating_dict = {}
item_ratings = ratings.groupby('movie_id')
for item in item_ratings:
    item_i_ratings = item[1][['user_id','rating']].set_index('user_id').to_dict()['rating']
    item_rating_dict[item[0]] = item_i_ratings
R = pd.DataFrame.from_dict(item_rating_dict, orient='index')
R = R.sort_index()
R = R.T

#将3.5分以上评分标为1
R35 = R.copy()
R35[R35<3.5]=0
R35[R35>=3.5]=1
R35 = R35.fillna(0)

#R*Q得到人-标签喜好度表
P = R35.dot(Q)

#通过标签向量对电影/人的描述得到人对电影的兴趣度表
c = cdist(P.values, Q.values, 'cosine')
S = pd.DataFrame(1 - c, index=P.index, columns=Q.index)

#获取推荐
def GetTopTenForUser(user_id):
    s = S.loc[user_id]
    s = s[s.notnull() & s != 0].sort_values()
    return s.nlargest(10)

recomm_movies = GetTopTenForUser(320)
recomm_movie_ids = recomm_movies.keys().tolist()
recomm_movie_similarty = recomm_movies.as_matrix()

recommendation_df_unweighted = pd.DataFrame(
    {'movie_id': recomm_movie_ids,
     'similarities': recomm_movie_similarty
    })

print("无权重")
print(recommendation_df_unweighted)

#改变人对电影的兴趣度计算方式，加入人的打分习惯考量与具体分数
RW = R.copy()
mu = RW.mean(axis=1)
W = RW.sub(mu, axis=0)
W = W.fillna(0)
PW = W.dot(Q)

cw = cdist(PW.values, Q.values, 'cosine')
SW = pd.DataFrame(1 - cw, index=PW.index, columns=Q.index)

def GetTopTenForUser_Weighted(user_id):
    sw = SW.loc[user_id]
    sw = sw[sw.notnull() & sw != 0].sort_values()
    return sw.nlargest(10)

recomm_weighted = GetTopTenForUser_Weighted(320)

recomm_movie_ids_weighted = recomm_weighted.keys().tolist()
recomm_movie_similarty_weighted = recomm_weighted.as_matrix()
recommendation_df_weighted = pd.DataFrame(
    {'movie_id': recomm_movie_ids_weighted,
     'similarities': recomm_movie_similarty_weighted
    })
print("有权重")
print(recommendation_df_weighted)





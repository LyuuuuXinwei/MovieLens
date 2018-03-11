# -*- coding:utf8 -*-

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
import random

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

'''
links--movieID,imdbID,tmdbID
movies--movieID,title,genres标签分类
ratings--userID,movieID,rating,timestamp
tags--userID,movieID,tag(评价),timestamp
'''

def parseRating(line):
    fields = line.strip().split(",")
    tail_num = fields[3] % 10
    return tail_num , (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    fields = line.strip().split(",")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    if not isfile(ratingsFile):
        print("%s 文件不存在" % ratingsFile)
        sys.exit(1)
    with open(ratingsFile, 'r') as f:
        ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])

    return ratings

def computeRmse(model, data, n):
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])).join(data.map(lambda x: ((x[0], x[1]), x[2]))).values()
    rmse = sqrt(predictionsAndRatings.mapvalues(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))
    return rmse


if __name__ == "__main__":
    
    conf = SparkConf().setAppName("MovieLensALS").set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    # 得到的ratings为(时间戳最后一位整数, (userId, movieId, rating))格式的RDD
    ratings = sc.textFile(join(data, "ratings.csv")).map(parseRating)

    # 得到的movies为(movieId, movieTitle)格式的RDD
    movies = dict(sc.textFile(join(data, "movies.csv")).map(parseMovie).collect())

    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

    # 根据时间戳最后一位把整个数据集分成训练集(60%), 交叉验证集(20%), 和评估集(20%)
    test_rmse_list = []
    validation_rmse_list = []
    parameter_list = []
    improvement_list = []
    numPartitions = 4
    iter_num = 5

    for i in range(iter_num):
        random_list = list(range(10))
        random.shuffle(random_list)
        training_list = random_list[:6]
        validation_list = random_list[6:8]
        test_list = random_list[8:]

        training = ratings.filter(lambda x: x in training_list).values().repartition(numPartitions).cache()
        validation = ratings.filter(lambda x: x in validation_list).values().repartition(numPartitions).cache()
        test = ratings.filter(lambda x: x in test_list).values().cache()

        numTraining = training.count()
        numValidation = validation.count()
        numTest = test.count()

        #参数
        ranks = [10,20,30]
        lambdas = [0.001,0.01,0.1,1]
        numIters = [10,]
        #alpha = 1.0 控制矩阵分解时，被观察到的“用户- 产品”交互相对没被观察到的交互的权重。

        bestModel = None
        bestValidationRmse = float("inf") #只是为了初始化一个float
        bestRank = 0
        bestLambda = -1.0

        #trainImplicit潜交互偏好非负
        for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
            model = ALS.train(training, rank, numIter, lmbda)
            validationRmse = computeRmse(model, validation, numValidation)


            if validationRmse < bestValidationRmse:
                bestModel = model
                bestValidationRmse = validationRmse
                bestRank = rank
                bestLambda = lmbda

        bestParameter = {'bestRank':bestRank
                         'bestLambda':bestLambda}
        parameter_list.append(bestParameter)

        testRmse = computeRmse(bestModel, test, numTest)
        test_rmse_list.append(testRmse)
        validation_rmse_list.append(bestValidationRmse)

        # baseline平均分
        meanRating = training.union(validation).map(lambda x: x[2]).mean()
        baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
        improvement = (baselineRmse - testRmse) / baselineRmse * 100

        improvement_list.append(improvement)

    test_rmse_list = []
    validation_rmse_list = []
    parameter_list = []
    improvement_list = []

    test_rmse_mean = sum(test_rmse_list) / len(test_rmse_list)
    validation_rmse_mean = sum(validation_rmse_list) / len(validation_rmse_list)
    improvement_mean = sum(improvement_list) / len(improvement_list)

    print('测试集平均MSE：{0}，验证集平均MSE：{2}，平均提升：{2}'.format(test_rmse_mean,validation_rmse_mean,improvement_mean))

    # 对所有用户进行推荐
    users_all = ratings.map(lambda x:x[1][0]).distinct().collect()
    movies_all = movies.map(lambda x:x[0]).distinct().collect()
    pairs = product(users_all,movies_all)
    candidates = sc.parallelize(pairs,numPartitions)
    predictions = bestModel.predictAll(candidates).collect()
    recommendations_5 = sorted(predictions, key=lambda x: x[2], reverse=True)[:5]

    sc.stop()

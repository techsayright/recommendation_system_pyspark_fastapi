from fastapi import FastAPI
from pyspark.sql import SparkSession
from pydantic import BaseModel
from pyspark.sql.functions import col, concat, lit, lower, udf, row_number, monotonically_increasing_id

from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.porter import PorterStemmer

from pyspark.sql.types import StringType

from sklearn.metrics.pairwise import cosine_similarity

import requests
from typing import List, Optional
ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return ' '.join(y)


stem_udf = udf(stem, StringType())

app = FastAPI()

input_uri = "mongodb+srv://username:password@globalplay-cluster-prep.zihzn.mongodb.net/globalplay-preprod?retryWrites=true&w=majority"
output_uri = "mongodb+srv://username:password@globalplay-cluster-prep.zihzn.mongodb.net/globalplay-preprod?retryWrites=true&w=majority"

spark2 = SparkSession.builder \
    .appName("myProject") \
    .config("spark.mongodb.input.uri", input_uri) \
    .config("spark.mongodb.input.collection", "users, categories")\
    .config("spark.mongodb.output.uri", output_uri) \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.2") \
    .getOrCreate()

df_read = spark2.read.format("com.mongodb.spark.sql.DefaultSource").option(
    "collection", "categories").load()

df_read_users = spark2.read.format(
    "com.mongodb.spark.sql.DefaultSource").option("collection", "users").load()


temp = []

category_data = df_read.select(col('_id.oid').alias('id'), 'categoryName')


users_data = df_read_users.select(col('_id.oid').alias(
    'location_id'), 'purveyorName', 'city', 'address', col('category.oid').alias('category_id'), 'description')

users_data = users_data.na.drop(how='any', thresh=None, subset=['category_id'])

location_data_joined = users_data.join(
    category_data, users_data.category_id == category_data.id, 'inner')

location_data_joined = location_data_joined.withColumn('tags', concat(col('purveyorName'), lit(
    ' '), col('address'), lit(' '), col('description'), lit(' '), col('categoryName')))

location_data_with_tag = location_data_joined.select(
    'location_id', 'purveyorName', 'categoryName', lower(col('tags')).alias('tags'))

location_data_with_tag = location_data_with_tag.withColumn(
    "row_id", monotonically_increasing_id())

# print('a')

cv = CountVectorizer(max_features=5000, stop_words='english')

# print('b')

tags = location_data_with_tag.select("tags").rdd.flatMap(lambda x: x).collect()

vectors = cv.fit_transform(tags).toarray()

# print('c')

location_data_with_tag = location_data_with_tag.withColumn(
    'new_tags', stem_udf(col('tags')))

similarity = cosine_similarity(vectors)

# print('d')


def recommend(location_id):
    global temp
    temp = []

    if (location_id is None):
        response = requests.post(
            'https://dashboard-preprod.funfull.com/api/queries/353/results?api_key=fhwOppGBknwssPNMDgjxf2wc2dYmR8MxuqKgAoTy')
        if response.status_code == 200:
            # print("sucessfully fetched the data")
            data = response.json()
            # print(data['query_result']['data']['rows'])
            # temp.append(data['query_result']['data']['rows'])
            temp = [item['id']
                    for item in data['query_result']['data']['rows']]
        else:
            print('api is not working')
    else:

        for id in location_id:
            loc_index = location_data_with_tag[location_data_with_tag['location_id'] == id].select(
                'row_id').collect()[0][0]
            distances = similarity[loc_index]
            loc_list = sorted(list(enumerate(distances)),
                              reverse=True, key=lambda x: x[1])[1:5]

            # print('e')

            for i in loc_list:
                temp.append(location_data_with_tag[location_data_with_tag['row_id'] == i[0]].select(
                    'location_id').collect()[0][0])
            

class req(BaseModel):
    # id: str
    id: Optional[List[str]] = None


@app.post("/recommend")
async def hello(data: req):
    # print(data)

    recommend(data.id)
    # recommend('null')
    return {"data": temp}

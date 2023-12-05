from fastapi import FastAPI
from pyspark.sql import SparkSession
from pydantic import BaseModel
from pyspark.sql.functions import col, concat, lit, lower, udf, monotonically_increasing_id
from pyspark.ml.feature import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
import requests
from nltk.stem.porter import PorterStemmer
from pyspark.sql.types import StringType
import numpy as np

ps = PorterStemmer()
stem_udf = udf(lambda x: ' '.join(ps.stem(word) for word in x.split()), StringType())

app = FastAPI()

input_uri = "mongodb+srv://data-warehouse-team-preprod:smsdDdddYqooID5aFTY4aNWFRuXnlL@globalplay-cluster-prep.zihzn.mongodb.net/globalplay-preprod?retryWrites=true&w=majority"
output_uri = "mongodb+srv://data-warehouse-team-preprod:smsdDdddYqooID5aFTY4aNWFRuXnlL@globalplay-cluster-prep.zihzn.mongodb.net/globalplay-preprod?retryWrites=true&w=majority"

spark = SparkSession.builder \
    .appName("myProject") \
    .config("spark.mongodb.input.uri", input_uri) \
    .config("spark.mongodb.input.collection", "users, categories") \
    .config("spark.mongodb.output.uri", output_uri) \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.2") \
    .getOrCreate()

df_read = spark.read.format("com.mongodb.spark.sql.DefaultSource").option(
    "collection", "categories").load()

df_read_users = spark.read.format(
    "com.mongodb.spark.sql.DefaultSource").option("collection", "users").load()

category_data = df_read.select(col('_id.oid').alias('id'), 'categoryName')

users_data = df_read_users.select(
    col('_id.oid').alias('location_id'),
    'purveyorName', 'city', 'address',
    col('category.oid').alias('category_id'), 'description'
).na.drop(how='any', thresh=None, subset=['category_id'])

location_data_joined = users_data.join(
    category_data, users_data.category_id == category_data.id, 'inner')

location_data_with_tag = location_data_joined.withColumn(
    'tags', concat(col('purveyorName'), lit(' '), col('address'), lit(' '), col('description'), lit(' '), col('categoryName'))
).withColumn('tags', lower(col('tags'))).withColumn(
    'new_tags', stem_udf(col('tags'))
).withColumn("row_id", monotonically_increasing_id())

cv = CountVectorizer(inputCol="new_tags", outputCol="features", maxFeatures=5000, minDF=2.0)

model = cv.fit(location_data_with_tag)
transformed_data = model.transform(location_data_with_tag)


# Collect the feature vectors as a NumPy array
vectors = transformed_data.select("features").rdd.map(lambda x: x.features.toArray()).collect()

# Convert the vectors to a NumPy array for sklearn compatibility
vectors_array = np.array(vectors)

# Calculate cosine similarity using sklearn
similarity = cosine_similarity(vectors_array)

def recommend(location_id):
    temp = []

    if location_id is None:
        # Fetch data using API
        response = requests.post(
            'https://dashboard-preprod.funfull.com/api/queries/353/results?api_key=fhwOppGBknwssPNMDgjxf2wc2dYmR8MxuqKgAoTy')
        if response.status_code == 200:
            data = response.json()
            temp = [item['id'] for item in data['query_result']['data']['rows']]
        else:
            print('API is not working')
    else:
        for loc_id in location_id:
            loc_index = transformed_data.filter(col('location_id') == loc_id).select(
                'row_id').collect()[0][0]
            distances = similarity[loc_index]
            loc_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:5]

            for i in loc_list:
                recommended_loc_id = location_data_with_tag.filter(col('row_id') == i[0]).select(
                    'location_id').collect()[0][0]
                temp.append(recommended_loc_id)

    return temp

class Request(BaseModel):
    id: Optional[List[str]] = None

@app.post("/recommend")
async def get_recommendations(data: Request):
    result = recommend(data.id)
    return {"data": result}

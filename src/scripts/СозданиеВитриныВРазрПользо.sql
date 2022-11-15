import findspark, csv
findspark.init()
findspark.find()
import os
os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['YARN_CONF_DIR'] = '/etc/hadoop/conf'
from pyspark.sql import SparkSession
import pyspark.sql.dataframe
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pyspark
from pyspark.sql.window import Window 
spark = SparkSession.builder \
                    .master("local") \
                    .getOrCreate()
path_event_prqt = "/user/master/data/geo/events"
path_city_data = "/user/tolmachevs/geo.csv"

def event_with_city(path_event_prqt: str, path_city_data: str, spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    events_geo = spark.read.parquet(path_event_prqt) \
            .sample(0.05) \
            .drop('city','id') \
            .withColumn('event_id', F.monotonically_increasing_id())
    df = spark.read.csv(path_city_data, sep = ";", header = True )
    city = df.withColumn('lat_1', F.split(df['lat'], ',').getItem(0)) \
             .withColumn('lat_2', F.split(df['lat'], ',').getItem(1)) \
             .withColumn('lng_1', F.split(df['lng'], ',').getItem(0)) \
             .withColumn('lng_2', F.split(df['lng'], ',').getItem(1)).drop('lat','lng') \
             .withColumn('diff', F.sqrt(F.sin((F.radians(F.col('lat_2')) - F.radians(F.col('lat_1')))/F.lit(2))+F.cos(F.radians(F.col('lat_1')))+F.cos(F.radians(F.col('lat_2')))*F.sin((F.radians(F.col('lng_2')) - F.radians(F.col('lng_1')))/F.lit(2)))*F.lit(6371)).persist()
    events_city = events_geo \
                .crossJoin(city).drop('lat_1','lat_2','lng_1','lng_2')
    return events_city
def event_corr_city(events_city: pyspark.sql.DataFrame, spark: pyspark.sql.SparkSession):
    window = Window().partitionBy('event.message_from').orderBy(F.col('diff').desc())
    df_city = events_city \
            .withColumn("row_number", F.row_number().over(window)) \
            .filter(F.col('row_number')==1) \
            .drop('row_number')
    return df_city
def actial_geo(df_city: pyspark.sql.DataFrame, spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    window = Window().partitionBy('event.message_from').orderBy(F.col('date').desc())
    df_actual = df_city \
        .withColumn("row_number", F.row_number().over(window)) \
        .filter(F.col('row_number')==1) \
        .selectExpr('event.message_from as user', 'city' , 'id as city_id') \
        .persist()
    return df_actual
	
def travel_geo(df_city: pyspark.sql.DataFrame, spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    window = Window().partitionBy('event.message_from', 'id').orderBy(F.col('date'))
    df_travel = df_city \
    .withColumn("dense_rank", F.dense_rank().over(window)) \
    .withColumn("date_diff", F.datediff(F.col('date').cast(DateType()), F.to_date(F.col("dense_rank").cast("string"), 'dd'))) \
    .selectExpr('date_diff', 'event.message_from as user', 'date', "id" ) \
    .groupBy("user", "date_diff", "id") \
    .agg(F.countDistinct(F.col('date')).alias('cnt_city'))
    return df_travel
	
def home_geo(df_travel: pyspark.sql.DataFrame, spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    df_home = df_travel \
    .withColumn('max_dt', F.max(F.col('date_diff')) \
                .over(Window().partitionBy('user')))\
    .filter((F.col('cnt_city')>27) & (F.col('date_diff') == F.col('max_dt'))) \
    .persist()
    return df_home
	
def DF_local_time(df_city: pyspark.sql.DataFrame, spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    DF_local_time = df_city.withColumn("TIME",F.col("event.datetime").cast("Timestamp"))\
    .withColumn("timezone",F.concat(F.lit("Australia"),F.col('city'))) \
    .withColumn("local_time",F.from_utc_timestamp(F.col("TIME"),F.col('timezone')))\
    .select("TIME", "local_time", 'city')
    return DF_local_time
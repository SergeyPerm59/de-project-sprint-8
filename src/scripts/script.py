import os

from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as f, DataFrame
from pyspark.sql.functions import from_json, to_json, col, lit, struct
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType


spark_jars_packages = ",".join(
        [
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0",
            "org.postgresql:postgresql:42.4.0",
        ]
    )

spark = SparkSession.builder \
    .appName("RestaurantSubscribeStreamingService") \
    .config("spark.sql.session.timeZone", "UTC") \
    .config("spark.jars.packages", spark_jars_packages) \
    .getOrCreate()

kafka_security_options = {
'kafka.bootstrap.servers':'rc1b-2erh7b35n4j4v869.mdb.yandexcloud.net:9091',
'kafka.security.protocol': 'SASL_SSL',
'kafka.sasl.mechanism': 'SCRAM-SHA-512',
'kafka.sasl.jaas.config': 'org.apache.kafka.common.security.scram.ScramLoginModule required username=\"de-student\" password=\"ltcneltyn\";',
"subscribe": 'tolmachev'
}

# читаем из топика Kafka сообщения с акциями от ресторанов 
restaurant_read_stream_df = spark.readStream \
.format('kafka') \
.option('kafka.bootstrap.servers', 'rc1b-2erh7b35n4j4v869.mdb.yandexcloud.net:9091') \
.option('kafka.security.protocol', 'SASL_SSL') \
.option('kafka.sasl.jaas.config', 'org.apache.kafka.common.security.scram.ScramLoginModule required username="de-student" password="ltcneltyn";') \
.option('kafka.sasl.mechanism', 'SCRAM-SHA-512') \
.option('subscribe', 'tolmachev') \
.load()
restaurant_read_stream_df.printSchema()

incomming_message_schema = StructType([
        StructField("restaurant_id", StringType(), nullable=True),
        StructField("adv_campaign_id", StringType(), nullable=True),
        StructField("adv_campaign_content", StringType(), nullable=True),
        StructField("adv_campaign_owner", StringType(), nullable=True),
        StructField("adv_campaign_owner_contact", StringType(), nullable=True),
        StructField("adv_campaign_datetime_start", LongType(), nullable=True),
        StructField("adv_campaign_datetime_end", LongType(), nullable=True),
        StructField("datetime_created", LongType(), nullable=True),
    ])


current_timestamp_utc = int(round(datetime.utcnow().timestamp()))

filtered_read_stream_df = restaurant_read_stream_df\
.select(col("value").cast(StringType()).alias("value_str"))\
.withColumn("deserialized_value", from_json(col("value_str"), schema=incomming_message_schema))\
.select("deserialized_value.*")\
.filter((col("adv_campaign_datetime_start") <= current_timestamp_utc) & (col("adv_campaign_datetime_end") >= current_timestamp_utc))

subscribers_restaurant_df = spark.read \
                    .format('jdbc') \
                    .option('url', 'jdbc:postgresql://rc1a-fswjkpli01zafgjm.mdb.yandexcloud.net:6432/de') \
                    .option('driver', 'org.postgresql.Driver') \
                    .option('dbtable', 'public.subscribers_restaurants') \
                    .option('user', 'student') \
                    .option('password', 'de-student') \
                    .load()


def foreach_batch_function(df, epoch_id):
    df.withColumn("feedback", f.lit(None).cast(StringType())).drop('id') \
       .write.format("jdbc") \
       .mode('append') \
       .option('url', 'jdbc:postgresql://localhost:5432/de') \
       .option('dbtable', 'public.subscribers_feedback') \
       .option('user', 'jovyan') \
       .option('password', 'jovyan') \
       .option('driver', 'org.postgresql.Driver') \
       .save()
    sd = df.withColumn('value', \
       f.to_json(f.struct(\
       f.col('restaurant_id'),\
       f.col('adv_campaign_id'),\
       f.col('adv_campaign_content'),\
       f.col('adv_campaign_owner'),\
       f.col('adv_campaign_owner_contact'),\
       f.col('adv_campaign_datetime_start'),\
       f.col('adv_campaign_datetime_end'),\
       f.col('datetime_created'),\
       f.col('id'),\
       f.col('client_id'),\
       f.col('trigger_datetime_created'))\
       )
       )\
       .select('value')
    sd.write\
        .format('kafka') \
        .option('kafka.bootstrap.servers', 'rc1b-2erh7b35n4j4v869.mdb.yandexcloud.net:9091') \
        .option('kafka.security.protocol', 'SASL_SSL') \
        .option('kafka.sasl.mechanism', 'SCRAM-SHA-512') \
        .option('kafka.ssl.truststore.location', '/usr/lib/jvm/java-1.17.0-openjdk-amd64/lib/security/cacerts') \
        .option('kafka.ssl.truststore.password', 'changeit') \
        .option('subscribe', 'tolmachevs') \
        .option("checkpointLocation", "pa")\
        .option("truncate", False)

subscribers_restaurant_df.show()

result_join = filtered_read_stream_df.alias('f').join(subscribers_restaurant_df.alias('g'), 'restaurant_id')\
    .withColumn("trigger_datetime_created", lit(int(round(datetime.utcnow().timestamp()))))\
    .dropDuplicates(['client_id', 'restaurant_id'])\
    .withWatermark('timestamp', '1 minutes')

result_join.printSchema()

result_join.write \
    .outputMode("append") \
    .trigger(processingTime="60 seconds") \
    .foreachBatch(foreach_batch_function) \
    .start() \
    .awaitTermination()
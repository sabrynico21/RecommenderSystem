
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, array_distinct
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import size
from dotenv import load_dotenv
from pyspark.sql.functions import array_contains
from pyspark.sql.functions import explode, col
import os
import pickle

def save_frequent_items(selected_nodes):
    spark = SparkSession.builder \
        .appName("Frequent Items Analysis") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .master("local[*]") \
        .config("spark.local.dir", "C:/Users/sabry/Documents/spark_temp") \
        .getOrCreate()
    frequent_items = []
    df = spark.read.parquet("../output/frequent_items")
    #df.show()
    for node in selected_nodes: 
        filtered_df = df.filter(array_contains(df.items, node))
        #filtered.show(truncate=False)
        exploded_df = filtered_df.select(explode(col("items")).alias("item"))
        distinct_items = exploded_df.select("item").distinct()
        unique_items = [row['item'] for row in distinct_items.collect()]
        frequent_items.append(unique_items)
        
    with open('../Results/Frequent Itemset Comparison/frequent_items.pkl', 'wb') as f: 
        pickle.dump(frequent_items, f)
    #print(frequent_items)
    spark.stop()

def calculate_frequent_items():
    load_dotenv()

    os.environ['JAVA_HOME'] = os.getenv('JAVA_HOME')
    os.environ['PYSPARK_PYTHON'] = os.getenv('PYSPARK_PYTHON')
    os.environ['SPARK_LOCAL_DIRS'] = os.getenv('SPARK_LOCAL_DIRS')
    os.environ['HADOOP_HOME'] = os.getenv('HADOOP_HOME')

    spark = SparkSession.builder \
        .appName("Market Basket Analysis") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .config("spark.executor.memory", "16g") \
        .config("spark.driver.memory", "16g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "8g") \
        .master("local[*]") \
        .config("spark.local.dir", "C:/Users/sabry/Documents/spark_temp") \
        .getOrCreate()

    df_raw = spark.read.csv(os.getenv('GROUPED_PRODUCTS'), header=True, inferSchema=True)
    df_raw.show()
    df = df_raw.withColumn("products", array_distinct(split(df_raw["products"], " ")))

    df = df.repartition(16)

    # Apply FPGrowth
    fpGrowth = FPGrowth(itemsCol="products", minSupport=0.0001, minConfidence=0.5, numPartitions=32)
    model = fpGrowth.fit(df)

    # Results
    print("Frequent Itemsets:")
    # model.freqItemsets \
    #     .filter(size("items") >= 2) \
    #     .show()

    filtered_itemsets = model.freqItemsets.filter(size("items") >= 2)

    filtered_itemsets.write \
        .format("parquet")  \
        .mode("overwrite") \
        .save("../output/frequent_items")


    print("Association Rules:")
    #model.associationRules.show()
    # model.associationRules.write \
    #     .format("parquet")  \
    #     .mode("overwrite") \
    #     .save("output/association_rules")

    print("Predictions:")
    #model.transform(df).show()
    # model.transform(df).write \
    #     .format("parquet")  \
    #     .mode("overwrite") \
    #     .save("output/predictions")

    spark.stop()
from pyspark.ml.feature import StringIndexer
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer


def type_indexer(data):
    stringIndexer = StringIndexer(inputCol="type", outputCol="typeIndexed")
    model = stringIndexer.fit(data)
    indexed = model.transform(data)
    indexed = indexed.drop('type')
    return indexed


def nameOrig_indexer(data):
    stringIndexer = StringIndexer(inputCol="str_orig", outputCol="nameOrigIndexed")
    model = stringIndexer.fit(data)
    indexed = model.transform(data)
    indexed = indexed.drop('str_orig')
    return indexed


def nameDest_indexer(data):
    stringIndexer = StringIndexer(inputCol="str_dest", outputCol="nameDestIndexed")
    model = stringIndexer.fit(data)
    indexed = model.transform(data)
    indexed = indexed.drop('str_dest')
    return indexed
    

def vectorize_fraud_data(indexed_data):   
    assembler = VectorAssembler(inputCols=['amount', 'newbalanceOrig', 'oldbalanceDest', 
                                       'num_orig', 'num_dest', 'typeIndexed','nameOrigIndexed',
                                       'nameDestIndexed'], outputCol='features')
    df_vector = assembler.transform(indexed_data)
    return df_vector


def correlation_matrix(df):
    matrix = Correlation.corr(df, 'features')
    return matrix.collect()[0]["pearson({})".format('features')].values


def create_numDf(df):
    df = df.withColumn("num_orig", df.num_orig.cast(DoubleType()))
    df = df.withColumn("num_dest", df.num_orig.cast(DoubleType()))
    return df
    

def sepNameOrig(df):
    df = (
    df.withColumn('str_orig', F.substring('nameOrig', 1,1))
    .withColumn('num_orig', F.col('nameOrig').substr(F.lit(2), F.length('nameOrig') - F.lit(1)))
)
    df = df.drop('nameOrig')
    return df


def sepNameDest(df):  
    df = (
        df.withColumn('str_dest', F.substring('nameDest', 1,1))
        .withColumn('num_dest', F.col('nameDest').substr(F.lit(2), F.length('nameDest') - F.lit(1)))
    )
    df = df.drop('nameDest')
    return df  


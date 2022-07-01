import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
//import org.apache.spark.ml.classification.{KNNClassifier, LogisticRegression}

object Cancer {
  def main(args: Array[String]): Unit = {

    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)
    //Creando el contexto del Servidor
    val sc = new SparkContext("local", "Salarios", System.getenv("SPARK_HOME"))
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("CargaJSON")
      .config("log4j.rootCategory", "ERROR, console")
      .getOrCreate()

    import spark.implicits._
    val df = spark.read.format("csv").option("header", "true")
      .option("inferSchema", "true").option("delimiter", ",").load("./resources/data.csv").toDF()

    df.show()

    val df1= df.drop("_c32", "id")
    //df1.write.format("csv").option("header", "true").option("delimeter", ",").save("resources/intermedioCancer")
    val inputColumns = Array("radius_mean", "texture_mean", "texture_mean", "area_mean", "smoothness_mean", "compactness_mean",  "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se",
      "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se","symmetry_se","fractal_dimension_se", "radius_worst" , "texture_worst", "perimeter_worst" , "area_worst", "smoothness_worst", "compactness_worst"
    ,"concavity_worst", "concave points_worst", "symmetry_worst","fractal_dimension_worst" )

    val assembler = new VectorAssembler().setInputCols(inputColumns)
      .setOutputCol("features")
    val featureSet = assembler.transform(df1)
    df1.limit(10).show()
    val labelIndexer = new StringIndexer().setInputCol("diagnosis").setOutputCol("label")
    val df3 = labelIndexer.fit(featureSet).transform(featureSet)


    /**
     *   Now we declare the LR model and run fit and transform to make predictions.
     */
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), 42)

    val model = new LogisticRegression().fit(trainingData)
    val predictions = model.transform(testData)
    predictions.select ("features", "label", "prediction").show()

      model.save("/Users/hubsantander/Desktop")
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")

    // measure the accuracy
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy del modelo  : "+ accuracy)

    /*val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(3)
      .setInputCol("features")
      .setOutputCol("diagnosis")*/
/*
    val knn = new KNNClassifier()
      .setTopTreeSize(df1.count().toInt / 500)
      .setFeaturesCol("features")
      .setPredictionCol("predicted")
      .setK(1)*/
/*
    val key = Vectors.dense(df1(1))

    val model = brp.fit(df1)

    // Feature Transformation
    model.transform(df1).show()
    // Cache the transformed columns
    val transformedA = model.transform(df1).cache()

    // Approximate similarity join
    model.approxSimilarityJoin(dfA, dfB, 1.5).show()
    model.approxSimilarityJoin(transformedA, transformedB, 1.5).show()
    // Self Join
    model.approxSimilarityJoin(dfA, dfA, 2.5).filter("datasetA.id < datasetB.id").show()

    // Approximate nearest neighbor search
    model.approxNearestNeighbors(df1, "diagnosis", 2).show()*/
  }
}
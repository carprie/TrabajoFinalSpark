import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, LinearRegression}
import org.apache.spark.ml.evaluation.{RankingEvaluator, RegressionEvaluator}
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.ml.evaluation.RegressionEvaluator

object Salarios {
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
    val df = spark.read.format("csv").option("header", "true")
      .option("inferSchema", "true").option("delimiter", ",").load("./resources/ds_salaries.csv").toDF()
   // df.show()
    //df.describe().show()
 // Transformacion de valores
    val result= df.withColumn("experience_level_N", when(col("experience_level")=== "EN",1)
                  .when(col("experience_level")==="MI", 2)
                  .when(col("experience_level")==="SE", 3)
                  .when(col("experience_level")==="EX",4))
     val result1= result.withColumn("company_size_N", when(col("company_size")==="S", 1)
                  .when(col("company_size")==="M", 2)
                  .when(col("company_size")==="L",3))

    result1.limit(10).show()

    val employment_type = new StringIndexer()
      .setInputCol("employment_type")
      .setOutputCol("employment_type_F")
      .fit(result1)
    val dfIndex1 = employment_type.transform(result1)

    val job_title = new StringIndexer()
      .setInputCol("job_title")
      .setOutputCol("job_title_F")
      .fit(dfIndex1)
    val dfIndex2 = job_title.transform(dfIndex1)

    val salary_currency = new StringIndexer()
      .setInputCol("salary_currency")
      .setOutputCol("salary_currency_F")
      .fit(dfIndex2)
    val dfIndex3 = salary_currency.transform(dfIndex2)


    val employee_residence = new StringIndexer()
      .setInputCol("employee_residence")
      .setOutputCol("employee_residence_F")
      .fit(dfIndex3)
    val dfIndex4 = employee_residence.transform(dfIndex3)

    val company_location = new StringIndexer()
      .setInputCol("company_location")
      .setOutputCol("company_location_F")
      .fit(dfIndex4)
    val dfIndex5 = company_location.transform(dfIndex4)

   // dfIndex5.write.format("csv").option("header", "true").option("delimeter", ",").save("resources/intermedioSalario")

    dfIndex5.limit(10).show()

    val df1= dfIndex5.drop("salary", "_c0","experience_level" ,"employment_type", "job_title", "salary_currency", "employee_residence", "company_location" ,"company_size")
    val inputColumns = Array("work_year", "experience_level_N", "employment_type_F", "job_title_F", "salary_currency_F", "employee_residence_F",  "company_location_F", "company_size_N")
    val assembler = new VectorAssembler().setInputCols(inputColumns)
      .setOutputCol("features")
    df1.limit(10).show()

    val featureSet = assembler.transform(df1)
    //val assembler2 = new StringIndexer().setInputCol("salary_in_usd").setOutputCol("label").fit(featureSet)
   // val featureSet1 = assembler2.transform(featureSet)
   // featureSet.write.format("csv").option("header", "true").option("delimeter", ",").save("resources/finalSalario")

    val Array(trainingData, testData) = featureSet.randomSplit(Array(0.7, 0.3), 42)

    val lr = new LinearRegression()
      .setLabelCol("salary_in_usd")
      .setFeaturesCol("features")
      .setMaxIter(1000)
      .setRegParam(0.9)
      .setElasticNetParam(0.3)
//, predictionCol="salary_in_usd"

    // Fit the model
    val lrModel = lr.fit(trainingData)
    val lrPredictions = lrModel.transform(testData)
    val test= lrPredictions.summary()
    lrModel.save("/Users/hubsantander/Desktop")
    val evaluator = new RegressionEvaluator()
    println("=====================================================")
    println(test.show())
    //println(evaluator.evaluate( lrPredictions).toString)
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary

    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

/*

    val featureSet = assembler.transform(df1)

    val labelInp = new StringIndexer().setInputCol("salary_in_usd_F").setInputCol("label")

   // val data= labelInp.fit(featureSet).transform(featureSet)

    val training = featureSet


    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
    */

val glr = new GeneralizedLinearRegression()
  .setFamily("gaussian")
  .setLink("identity")
  .setMaxIter(10)
  .setRegParam(0.3)
  .setLabelCol("salary_in_usd")
  .setFeaturesCol("features")

    // Fit the model
    val gmodel = glr.fit(trainingData)
   val glrPredictions = gmodel.transform(testData)
   val gtest= glrPredictions.summary()

   println("=====================================================")
   println(gtest.show())

    println(s"Coefficients: ${gmodel.coefficients}")
    println(s"Intercept: ${gmodel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val gsummary = gmodel.summary
    println(s"Coefficient Standard Errors g: ${gsummary.coefficientStandardErrors.mkString(",")}")
    println(s"T Values g: ${gsummary.tValues.mkString(",")}")
    println(s"P Values g: ${gsummary.pValues.mkString(",")}")
    println(s"Dispersion g: ${gsummary.dispersion}")
    println(s"Null Deviance g: ${gsummary.nullDeviance}")
    println(s"Residual Degree Of Freedom Null g: ${gsummary.residualDegreeOfFreedomNull}")
    println(s"Deviance g: ${gsummary.deviance}")
    println(s"Residual Degree Of Freedom: ${gsummary.residualDegreeOfFreedom}")
    println(s"AIC g: ${gsummary.aic}")
    println("Deviance Residuals g: ")
   println(s"r2 g: ${gsummary }")
    gsummary.residuals().show()

    //println(s"r2: ${gsummary.r2 }")
   s"r2: ${trainingSummary.r2}"



  }
}

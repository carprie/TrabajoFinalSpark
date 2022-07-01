ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.8"

lazy val root = (project in file("."))
  .settings(
    name := "TrabajoFinal"
  )


val sparkVersion = "3.3.0"



// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
// libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion % "provided"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib

libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
//libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"


//libraryDependencies += "org.apache.spark" % "spark-streaming_2.11" % sparkVersion
libraryDependencies += "org.scalanlp" %% "breeze" % "1.1"
libraryDependencies += "org.scalanlp" %% "breeze-viz" % "1.1"
resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"


resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"

addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.6")
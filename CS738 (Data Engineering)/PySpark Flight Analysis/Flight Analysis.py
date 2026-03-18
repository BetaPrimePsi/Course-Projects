from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

def main(input_path):
    spark = SparkSession.builder.appName("FlightAnalysis").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Derive output base from input path (e.g. /user/$USER)
    import os
    user = os.environ.get("USER", "student")
    base_out = f"/user/{user}"

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Cast columns that may have been read as strings
    df = (df
        .withColumn("DEP_DELAY",       F.col("DEP_DELAY").cast("float"))
        .withColumn("ARR_DELAY",       F.col("ARR_DELAY").cast("float"))
        .withColumn("CANCELLED",       F.col("CANCELLED").cast("integer"))
        .withColumn("CARRIER_DELAY",   F.col("CARRIER_DELAY").cast("float"))
        .withColumn("WEATHER_DELAY",   F.col("WEATHER_DELAY").cast("float"))
        .withColumn("NAS_DELAY",       F.col("NAS_DELAY").cast("float"))
        .withColumn("FL_DATE",         F.col("FL_DATE").cast("string"))
    )

    # ------------------------------------------------------------------ #
    # Task 1 — Average Departure Delay per Airline (non-cancelled only)
    # Sort descending by average delay
    # ------------------------------------------------------------------ #
    task1_out = f"{base_out}/flight-task1"

    task1 = (df
        .filter(F.col("CANCELLED") == 0)
        .filter(F.col("DEP_DELAY").isNotNull())
        .groupBy("OP_CARRIER")
        .agg(F.round(F.avg("DEP_DELAY"), 2).alias("avg_dep_delay"))
        .orderBy(F.col("avg_dep_delay").desc())
    )

    task1.write.mode("overwrite").csv(task1_out, header=False)
    print(f"Task 1 written to {task1_out}")

    # ------------------------------------------------------------------ #
    # Task 2 — Top 10 Most Delayed Routes by Average Arrival Delay
    # ------------------------------------------------------------------ #
    task2_out = f"{base_out}/flight-task2"

    task2 = (df
        .filter(F.col("CANCELLED") == 0)
        .filter(F.col("ARR_DELAY").isNotNull())
        .withColumn("ROUTE", F.concat_ws("-", F.col("ORIGIN"), F.col("DEST")))
        .groupBy("ROUTE")
        .agg(F.round(F.avg("ARR_DELAY"), 2).alias("avg_arr_delay"))
        .orderBy(F.col("avg_arr_delay").desc())
        .limit(10)
    )

    task2.write.mode("overwrite").csv(task2_out, header=False)
    print(f"Task 2 written to {task2_out}")

    # ------------------------------------------------------------------ #
    # Task 3 — Cancellation Rate by Month (chronological order)
    # ------------------------------------------------------------------ #
    task3_out = f"{base_out}/flight-task3"

    task3 = (df
        .withColumn("MONTH", F.month(F.to_date(F.col("FL_DATE"), "yyyy-MM-dd")))
        .filter(F.col("MONTH").isNotNull())
        .groupBy("MONTH")
        .agg(
            F.round(
                F.sum(F.col("CANCELLED").cast("float")) / F.count("*") * 100,
                2
            ).alias("cancellation_rate")
        )
        .orderBy("MONTH")
    )

    task3.write.mode("overwrite").csv(task3_out, header=False)
    print(f"Task 3 written to {task3_out}")

    # ------------------------------------------------------------------ #
    # Task 4 — Primary Cause of Delay per Airline
    # Only rows where at least one delay column is non-zero and non-null
    # ------------------------------------------------------------------ #
    task4_out = f"{base_out}/flight-task4"

    task4_base = (df
        .filter(
            (F.col("CARRIER_DELAY").isNotNull() & (F.col("CARRIER_DELAY") != 0)) |
            (F.col("WEATHER_DELAY").isNotNull() & (F.col("WEATHER_DELAY") != 0)) |
            (F.col("NAS_DELAY").isNotNull()     & (F.col("NAS_DELAY")     != 0))
        )
        .groupBy("OP_CARRIER")
        .agg(
            F.round(F.sum(F.coalesce(F.col("CARRIER_DELAY"), F.lit(0))), 2).alias("total_carrier"),
            F.round(F.sum(F.coalesce(F.col("WEATHER_DELAY"), F.lit(0))), 2).alias("total_weather"),
            F.round(F.sum(F.coalesce(F.col("NAS_DELAY"),     F.lit(0))), 2).alias("total_nas"),
        )
    )

    # Pick whichever of the three totals is largest
    task4 = (task4_base
        .withColumn("dominant_cause",
            F.when(
                (F.col("total_carrier") >= F.col("total_weather")) &
                (F.col("total_carrier") >= F.col("total_nas")),
                F.lit("CARRIER_DELAY")
            ).when(
                F.col("total_weather") >= F.col("total_nas"),
                F.lit("WEATHER_DELAY")
            ).otherwise(F.lit("NAS_DELAY"))
        )
        .withColumn("total_minutes",
            F.when(F.col("dominant_cause") == "CARRIER_DELAY", F.col("total_carrier"))
             .when(F.col("dominant_cause") == "WEATHER_DELAY", F.col("total_weather"))
             .otherwise(F.col("total_nas"))
        )
        .select("OP_CARRIER", "dominant_cause", "total_minutes")
        .orderBy("OP_CARRIER")
    )

    task4.write.mode("overwrite").csv(task4_out, header=False)
    print(f"Task 4 written to {task4_out}")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: spark-submit flight_analysis.py <input_path>")
        sys.exit(1)
    main(sys.argv[1])

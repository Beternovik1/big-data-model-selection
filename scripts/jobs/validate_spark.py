"""Validate distributed computation through Apache Spark"""
import logging
from pyspark.sql import SparkSession

# constants
APPLICATION_NAME = "DistributedSumValidation"
PARTITION_COUNT = 2
RANGE_START = 1
RANGE_STOP = 101
EXPECTED_SUM = 5050

LOGGER = logging.getLogger(__name__)


def calculate_sum(spark: SparkSession) -> int:
    """Calculate an integer sum using Spark partitions"""
    numbers = spark.sparkContext.parallelize(
        range(RANGE_START, RANGE_STOP),
        PARTITION_COUNT,
    )
    return int(numbers.sum())

def main() -> None:
    """Run and validate the Spark computation"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    spark = SparkSession.builder.appName(APPLICATION_NAME).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        result = calculate_sum(spark)
        if result != EXPECTED_SUM:
            raise RuntimeError(
                f"Expected {EXPECTED_SUM}, but Spark returned {result}"
            )
        LOGGER.info("Validation succeeded")
        LOGGER.info("Result: %d", result)
        LOGGER.info("Master: %s", spark.sparkContext.master)
        LOGGER.info("Partitions: %d", PARTITION_COUNT)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
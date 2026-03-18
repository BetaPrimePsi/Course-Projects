# CS 738 – Assignment 2: Distributed Flight Analysis (PySpark)

Analysis of U.S. domestic flight on-time performance data using PySpark on the UW CS datasci cluster. All four tasks are implemented in a single script.

## Dataset

Pre-loaded on the cluster at `/data/flights.csv`. Each row is a scheduled domestic flight with columns including airline code, origin/destination, departure/arrival delays, cancellation status, and delay cause breakdowns.

## Setup

```bash
# SSH into the cluster (VPN required)
ssh <watid>@datasci-login.cs.uwaterloo.ca

# Upload the script
scp flight_analysis.py <watid>@datasci-login.cs.uwaterloo.ca:~/

# Clear old output directories
hdfs dfs -rm -r -f /user/$USER/flight-task1
hdfs dfs -rm -r -f /user/$USER/flight-task2
hdfs dfs -rm -r -f /user/$USER/flight-task3
hdfs dfs -rm -r -f /user/$USER/flight-task4
```

## Running

```bash
spark-submit flight_analysis.py /data/flights.csv
```

All four tasks run in a single job. Output is written to HDFS under `/user/$USER/flight-taskN/`.

---

## Tasks

### Task 1 — Average Departure Delay per Airline

Filters to non-cancelled flights with valid departure delay values, groups by airline code (`OP_CARRIER`), and computes the average departure delay in minutes. Sorted descending so the worst-performing airline appears first.

**Output:**
```
[TO BE FILLED AFTER CLUSTER RUN]
```

---

### Task 2 — Top 10 Most Delayed Routes

A route is defined as an `ORIGIN-DEST` pair. Filters to non-cancelled flights, computes average arrival delay per route, and returns the 10 routes with the highest average.

**Output:**
```
[TO BE FILLED AFTER CLUSTER RUN]
```

---

### Task 3 — Cancellation Rate by Month

Extracts the month from `FL_DATE`, then for each month computes the percentage of flights marked as cancelled. Output is in chronological order (1–12).

**Output:**
```
[TO BE FILLED AFTER CLUSTER RUN]
```

---

### Task 4 — Primary Cause of Delay per Airline

For flights with at least one non-zero delay cause, sums `CARRIER_DELAY`, `WEATHER_DELAY`, and `NAS_DELAY` per airline and identifies which category accounts for the most total minutes. Missing values are treated as zero.

**Output:**
```
[TO BE FILLED AFTER CLUSTER RUN]
```

---

## Viewing Output on Cluster

```bash
hdfs dfs -cat /user/$USER/flight-task1/part-*
hdfs dfs -cat /user/$USER/flight-task2/part-*
hdfs dfs -cat /user/$USER/flight-task3/part-*
hdfs dfs -cat /user/$USER/flight-task4/part-*
```

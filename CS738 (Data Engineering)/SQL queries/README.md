# CS 778 – Assignment 1: SQL Queries

MySQL queries written against the `Assignment_One` schema, covering joins, aggregations, subqueries, set operations, and DML.

## Setup

```bash
# Start the MySQL container
docker compose up -d

# Initialize the database
docker compose exec -it -w /cs738 mysql bash
mysql -u root -p < create.sql
```

## Running Queries

From inside the MySQL shell:
```sql
USE Assignment_One;
source queries/1.sql
```

---

## Queries & Output

### Query 1 — Rooms used for more than 2 classes (sorted by class count desc)

```sql
-- queries/1.sql
```

```
+----------+-------------+
| room     | class_count |
+----------+-------------+
| R15      |           5 |
| R128     |           5 |
| 20 AVW   |           4 |
| 1320 DCL |           3 |
+----------+-------------+
4 rows in set (0.01 sec)
```

---

### Query 2 — Faculty members who teach no classes (sorted by name asc)

```sql
-- queries/2.sql
```

```
+----------------+
| fname          |
+----------------+
| David Anderson |
| James Smith    |
| Michael Miller |
| Ulysses Teach  |
+----------------+
4 rows in set (0.00 sec)
```

---

### Query 3 — Students enrolled in both 'Operating System Design' and 'Database Systems'

```sql
-- queries/3.sql
```

```
+--------------------+
| sname              |
+--------------------+
| Ana Lopez          |
| Christopher Garcia |
| Joseph Thompson    |
| Lisa Walker        |
| Lisa Walker        |
+--------------------+
5 rows in set (0.00 sec)
```

---

### Query 4 — All classes with lecturer name and enrollment size (sorted by size desc)

```sql
-- queries/4.sql
```

```
+---------------------------------+------------------+------------+
| class_name                      | lecturer_name    | class_size |
+---------------------------------+------------------+------------+
| Operating System Design         | Linda Davis      |          7 |
| Database Systems                | Ivana Teach      |          6 |
| Optical Electronics             | Patricia Jones   |          1 |
| American Political Parties      | Jennifer Thomas  |          1 |
| Data Structures                 | Linda Davis      |          1 |
| Urban Economics                 | Richard Jackson  |          1 |
| Perception                      | Richard Jackson  |          1 |
| Air Quality Engineering         | John Williams    |          1 |
| Social Cognition                | William Moore    |          1 |
| Communication Networks          | Mary Johnson     |          1 |
| Patent Law                      | Elizabeth Taylor |          1 |
| Archaeology of the Incas        | Barbara Wilson   |          0 |
| Introductory Latin              | Barbara Wilson   |          0 |
| Dairy Herd Management           | Robert Brown     |          0 |
| Intoduction to Math             | Richard Jackson  |          0 |
| Marketing Research              | Richard Jackson  |          0 |
| Organic Chemistry               | Richard Jackson  |          0 |
| Seminar in American Art         | Richard Jackson  |          0 |
| Multivariate Analysis           | Elizabeth Taylor |          0 |
| Orbital Mechanics               | John Williams    |          0 |
| Aviation Accident Investigation | John Williams    |          0 |
+---------------------------------+------------------+------------+
21 rows in set (0.00 sec)
```

---

### Query 5 — Number of students enrolled in exactly one class of size < 5

```sql
-- queries/5.sql
```

```
+---------------+
| student_count |
+---------------+
|             6 |
+---------------+
1 row in set (0.00 sec)
```

---

### Query 6 — Delete classes enrolled by fewer than 2 students

```sql
-- queries/6.sql
```

```
Query OK, 9 rows affected (0.01 sec)
```

---

### Query 7 — Freshmen students (standing = FR) sorted by age then name

```sql
-- queries/7.sql
```

```
+-----------+----------------+------+
| snum      | sname          | age  |
+-----------+----------------+------+
| 320874981 | Daniel Lee     |   17 |
| 455798411 | Luis Hernandez |   17 |
| 318548912 | Dorthy Lewis   |   18 |
| 567354612 | Karen Scott    |   18 |
| 280158572 | Margaret Clark |   18 |
| 451519864 | Mark Young     |   18 |
+-----------+----------------+------+
6 rows in set (0.00 sec)
```

---

### Query 8 — All classes with meeting time, room, lecturer, and department (sorted by class name)

```sql
-- queries/8.sql
```

```
+---------------------------------+------------------+----------+------------------+--------+
| name                            | meets_at         | room     | fname            | deptid |
+---------------------------------+------------------+----------+------------------+--------+
| Air Quality Engineering         | TuTh 10:30-11:45 | R15      | John Williams    |     68 |
| American Political Parties      | TuTh 2-3:15      | 20 AVW   | Jennifer Thomas  |     11 |
| Archaeology of the Incas        | MWF 3-4:15       | R128     | Barbara Wilson   |     12 |
| Aviation Accident Investigation | TuTh 1-2:50      | Q3       | John Williams    |     68 |
| Communication Networks          | MW 9:30-10:45    | 20 AVW   | Mary Johnson     |     20 |
| Dairy Herd Management           | TuTh 12:30-1:45  | R128     | Robert Brown     |     12 |
| Data Structures                 | MWF 10           | R128     | Linda Davis      |     20 |
| Database Systems                | MWF 12:30-1:45   | 1320 DCL | Ivana Teach      |     20 |
| Intoduction to Math             | TuTh 8-9:30      | R128     | Richard Jackson  |     33 |
| Introductory Latin              | MWF 3-4:15       | R12      | Barbara Wilson   |     12 |
| Marketing Research              | MW 10-11:15      | 1320 DCL | Richard Jackson  |     33 |
| Multivariate Analysis           | TuTh 2-3:15      | R15      | Elizabeth Taylor |     11 |
| Operating System Design         | TuTh 12-1:20     | 20 AVW   | Linda Davis      |     20 |
| Optical Electronics             | TuTh 12:30-1:45  | R15      | Patricia Jones   |     68 |
| Orbital Mechanics               | MWF 8            | 1320 DCL | John Williams    |     68 |
| Organic Chemistry               | TuTh 12:30-1:45  | R12      | Richard Jackson  |     33 |
| Patent Law                      | F 1-2:50         | R128     | Elizabeth Taylor |     11 |
| Perception                      | MTuWTh 3         | Q3       | Richard Jackson  |     33 |
| Seminar in American Art         | M 4              | R15      | Richard Jackson  |     33 |
| Social Cognition                | Tu 6:30-8:40     | R15      | William Moore    |     33 |
| Urban Economics                 | MWF 11           | 20 AVW   | Richard Jackson  |     33 |
+---------------------------------+------------------+----------+------------------+--------+
21 rows in set (0.00 sec)
```

---

### Query 9 — Classrooms hosting classes at more than one distinct meeting time

```sql
-- queries/9.sql
```

```
+----------+-------------+-----------+
| room     | num_classes | num_times |
+----------+-------------+-----------+
| R128     |           5 |         5 |
| R15      |           5 |         5 |
| 20 AVW   |           4 |         4 |
| 1320 DCL |           3 |         3 |
| Q3       |           2 |         2 |
| R12      |           2 |         2 |
+----------+-------------+-----------+
6 rows in set (0.00 sec)
```

---

### Query 10 — Majors with at least 2 students: count and average age

```sql
-- queries/10.sql
```

```
+------------------------+--------------+---------+
| major                  | num_students | avg_age |
+------------------------+--------------+---------+
| Computer Science       |            5 |   18.20 |
| Computer Engineering   |            2 |   18.50 |
| Electrical Engineering |            2 |   17.00 |
| Finance                |            2 |   18.00 |
| Psychology             |            2 |   19.00 |
+------------------------+--------------+---------+
5 rows in set (0.00 sec)
```

---

### Query 11 — Students not enrolled in any class (sorted by name)

```sql
-- queries/11.sql
```

```
+-----------+-----------------+
| snum      | sname           |
+-----------+-----------------+
| 132977562 | Angela Martinez |
| 574489456 | Betty Adams     |
|  60839453 | Charles Harris  |
| 320874981 | Daniel Lee      |
| 462156489 | Donald King     |
| 318548912 | Dorthy Lewis    |
| 578875478 | Edward Baker    |
| 550156548 | George Wright   |
| 301221823 | Juan Rodriguez  |
| 556784565 | Kenneth Hill    |
| 280158572 | Margaret Clark  |
|  51135593 | Maria White     |
| 451519864 | Mark Young      |
| 351565322 | Nancy Allen     |
| 573284895 | Steven Green    |
|  99354543 | Susan Martin    |
| 269734834 | Thomas Robinson |
+-----------+-----------------+
17 rows in set (0.00 sec)
```

---

### Query 12 — Each faculty member's classes taught and total enrollments

```sql
-- queries/12.sql
```

```
+-----------+------------------+----------------+-------------------+
| fid       | fname            | classes_taught | total_enrollments |
+-----------+------------------+----------------+-------------------+
| 489221823 | Richard Jackson  |              6 |                 0 |
|  11564812 | John Williams    |              3 |                 0 |
| 489456522 | Linda Davis      |              2 |                 7 |
| 248965255 | Barbara Wilson   |              2 |                 0 |
|  90873519 | Elizabeth Taylor |              2 |                 0 |
| 142519864 | Ivana Teach      |              1 |                 6 |
| 619023588 | Jennifer Thomas  |              1 |                 0 |
| 141582651 | Mary Johnson     |              1 |                 0 |
| 254099823 | Patricia Jones   |              1 |                 0 |
| 356187925 | Robert Brown     |              1 |                 0 |
| 159542516 | William Moore    |              1 |                 0 |
| 486512566 | David Anderson   |              0 |                 0 |
| 242518965 | James Smith      |              0 |                 0 |
| 287321212 | Michael Miller   |              0 |                 0 |
| 548977562 | Ulysses Teach    |              0 |                 0 |
+-----------+------------------+----------------+-------------------+
15 rows in set (0.00 sec)
```

---

### Query 13 — Pairs of students sharing at least 2 classes

```sql
-- queries/13.sql
```

```
+--------------------+--------------------+----------------+
| student1           | student2           | shared_classes |
+--------------------+--------------------+----------------+
| Christopher Garcia | Ana Lopez          |              2 |
| Christopher Garcia | Lisa Walker        |              2 |
| Joseph Thompson    | Ana Lopez          |              2 |
| Joseph Thompson    | Christopher Garcia |              2 |
| Joseph Thompson    | Lisa Walker        |              2 |
| Lisa Walker        | Ana Lopez          |              2 |
| Lisa Walker        | Ana Lopez          |              2 |
| Lisa Walker        | Christopher Garcia |              2 |
| Lisa Walker        | Joseph Thompson    |              2 |
| Lisa Walker        | Lisa Walker        |              2 |
+--------------------+--------------------+----------------+
10 rows in set (0.00 sec)
```

---

### Query 14 — Classes with enrollment below average (including zero-enrollment classes)

```sql
-- queries/14.sql
```

```
+-------------------------+------+
| cname                   | size |
+-------------------------+------+
| Operating System Design |    7 |
| Database Systems        |    6 |
+-------------------------+------+
2 rows in set (0.00 sec)
```

---

### Query 15 — Students enrolled in every class taught by 'Ivana Teach'

```sql
-- queries/15.sql
```

```
+--------------------+
| sname              |
+--------------------+
| Ana Lopez          |
| Christopher Garcia |
| Joseph Thompson    |
| Lisa Walker        |
| Lisa Walker        |
| Paul Hall          |
+--------------------+
6 rows in set (0.01 sec)
```

---

### Query 16 — Students with at least one class from both 'Ivana Teach' and 'Linda Davis'

```sql
-- queries/16.sql
```

```
+--------------------+
| sname              |
+--------------------+
| Ana Lopez          |
| Christopher Garcia |
| Joseph Thompson    |
| Lisa Walker        |
| Lisa Walker        |
+--------------------+
5 rows in set (0.00 sec)
```

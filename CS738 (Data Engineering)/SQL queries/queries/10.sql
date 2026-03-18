SELECT major,
        COUNT(*) AS num_students,
        ROUND(AVG(age), 2) AS avg_age
FROM student
GROUP BY major
HAVING COUNT(*) >= 2
ORDER BY num_students DESC, major ASC;
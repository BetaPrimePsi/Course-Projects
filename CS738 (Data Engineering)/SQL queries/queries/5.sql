SELECT COUNT(*) AS student_count
FROM (
    SELECT e.snum
    FROM enrolled e
    WHERE e.cname IN (
        SELECT cname
        FROM enrolled
        GROUP BY cname
        HAVING COUNT(*) < 5
    )
    GROUP BY e.snum
    HAVING COUNT(*) = 1
) AS targeted_students;
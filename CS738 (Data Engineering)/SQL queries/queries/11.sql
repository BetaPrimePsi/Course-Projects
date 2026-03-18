SELECT snum, sname
FROM student
WHERE snum NOT IN (
    SELECT snum
    FROM enrolled
)
ORDER BY sname ASC;
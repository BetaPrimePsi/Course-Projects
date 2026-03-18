DELETE FROM enrolled
WHERE cname IN (
    SELECT cname
    FROM (
        SELECT name AS cname
        FROM class c
        LEFT JOIN enrolled e ON c.name = e.cname
        GROUP BY c.name
        HAVING COUNT(e.snum) < 2
    ) AS enrolls
);
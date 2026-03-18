SELECT c.name AS cname, COUNT(e.snum) AS size
FROM class c
LEFT JOIN enrolled e ON c.name = e.cname
GROUP BY c.name
HAVING size > (
    SELECT AVG(cs2.size)
    FROM (
        SELECT COUNT(e.snum) AS size
        FROM class c
        LEFT JOIN enrolled e ON c.name = e.cname
        GROUP BY c.name
    ) cs2
)
ORDER BY size DESC, cname ASC;
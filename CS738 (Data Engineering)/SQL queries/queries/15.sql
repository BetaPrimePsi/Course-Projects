SELECT s.sname
FROM student s
JOIN enrolled e ON s.snum = e.snum
JOIN class c ON e.cname = c.name
JOIN faculty f ON c.fid = f.fid
WHERE f.fname = 'Ivana Teach'
GROUP BY s.snum, s.sname
HAVING COUNT(*) = (
    SELECT COUNT(*)
    FROM class c2
    JOIN faculty f2 ON c2.fid = f2.fid
    WHERE f2.fname = 'Ivana Teach'
)
ORDER BY s.sname ASC;
SELECT s.sname
FROM student s
JOIN enrolled e1 ON s.snum = e1.snum
JOIN enrolled e2 ON s.snum = e2.snum
WHERE e1.cname = 'Operating System Design'
    AND e2.cname = 'Database Systems'
ORDER BY s.sname ASC;
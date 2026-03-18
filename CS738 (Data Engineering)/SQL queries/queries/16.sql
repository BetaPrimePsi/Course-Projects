SELECT s.sname
FROM student s
WHERE EXISTS (
    SELECT 1
    FROM enrolled e
    JOIN class c ON e.cname = c.name
    JOIN faculty f ON c.fid = f.fid
    WHERE e.snum = s.snum
        AND f.fname = 'Ivana Teach'
)
AND EXISTS (
    SELECT 1
    FROM enrolled e
    JOIN class c ON e.cname = c.name
    JOIN faculty f ON c.fid = f.fid
    WHERE e.snum = s.snum
        AND f.fname = 'Linda Davis'
)
ORDER BY s.sname ASC;
SELECT s1.sname AS student1,
        s2.sname AS student2,
        COUNT(*) AS shared_classes
FROM enrolled e1
JOIN enrolled e2
    ON e1.cname = e2.cname
    AND e1.snum < e2.snum
JOIN student s1 ON s1.snum = e1.snum
JOIN student s2 ON s2.snum = e2.snum
GROUP BY e1.snum, s1.sname, e2.snum, s2.sname
HAVING COUNT(*) >= 2
ORDER BY shared_classes DESC, student1 ASC, student2 ASC;
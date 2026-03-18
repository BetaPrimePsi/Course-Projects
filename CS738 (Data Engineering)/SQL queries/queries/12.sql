SELECT f.fid,
    f.fname,
    COUNT(DISTINCT c.name) AS classes_taught,
    COUNT(e.snum) AS total_enrollments
FROM faculty f
LEFT JOIN class c ON f.fid = c.fid
LEFT JOIN enrolled e ON c.name = e.cname
GROUP BY f.fid, f.fname
ORDER BY classes_taught DESC,
        total_enrollments DESC,
        f.fname ASC;
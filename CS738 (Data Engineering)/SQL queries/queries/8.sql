SELECT c.name,
    c.meets_at,
    c.room,
    f.fname,
    f.deptid
FROM class c
JOIN faculty f ON c.fid = f.fid
ORDER BY c.name ASC;
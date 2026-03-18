SELECT fname
FROM faculty
WHERE fid NOT IN (
    SELECT DISTINCT fid
    FROM class
    WHERE fid IS NOT NULL
)
ORDER BY fname ASC;
SELECT room,
        COUNT(*) AS num_classes,
        COUNT(DISTINCT meets_at) AS num_times
FROM class
GROUP BY room
HAVING COUNT(DISTINCT meets_at) > 1
ORDER BY num_times DESC, num_classes DESC, room ASC;
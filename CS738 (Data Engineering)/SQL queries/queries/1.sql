SELECT room, COUNT(*) AS class_count
FROM class
GROUP BY room
HAVING COUNT(*) > 2
ORDER BY class_count DESC;
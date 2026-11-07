-- Integer UTC seconds; the lower boundary is inclusive, current timestamp exclusive.
-- An index supports the correlated range lookup. Same-time peers cannot leak in.
WITH history AS (
    SELECT t.*,
           (SELECT AVG(h.amount_cents) FROM transactions h
            WHERE h.user_id=t.user_id AND h.ts >= t.ts-2592000 AND h.ts < t.ts) AS past_mean_cents
    FROM transactions t
)
SELECT *, CASE WHEN past_mean_cents IS NOT NULL
                    AND amount_cents > 3*past_mean_cents THEN 1 ELSE 0 END AS is_anomaly
FROM history ORDER BY transaction_id;

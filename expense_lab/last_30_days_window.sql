-- SQLite analytic version, verified against the indexed reference query.
-- ts is INTEGER UTC seconds, so 1 PRECEDING excludes every same-time peer.
-- For subsecond timestamps, use the warehouse's time-aware boundary semantics.
WITH history AS (
    SELECT *, AVG(amount_cents) OVER (
        PARTITION BY user_id ORDER BY ts
        RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
    ) AS past_mean_cents
    FROM transactions
)
SELECT *, CASE WHEN past_mean_cents IS NOT NULL
                    AND amount_cents > 3*past_mean_cents THEN 1 ELSE 0 END AS is_anomaly
FROM history ORDER BY transaction_id;

-- Assume mem=200GB
ALTER SYSTEM SET shared_buffers = '50GB'; -- The value should be set to 15% to 25% of the machine’s total RAM.
ALTER SYSTEM SET work_mem = '1GB'; -- Total RAM * 0.25 / max_connections
ALTER SYSTEM SET maintenance_work_mem = '10GB'; -- Total RAM * 0.05
ALTER SYSTEM SET effective_cache_size = '100GB'; -- Recommendations are to set Effective_cache_size at 50% of the machine’s total RAM.
ALTER SYSTEM SET temp_buffers = '1.5GB';
-- ALTER SYSTEM RESET ALL;
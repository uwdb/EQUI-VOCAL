-- DB Version: 12
-- OS Type: linux
-- DB Type: web
-- Total Memory (RAM): 100 GB
-- CPUs num: 1
-- Data Storage: ssd

ALTER SYSTEM SET
 max_connections = '20';
ALTER SYSTEM SET
 shared_buffers = '25GB';
ALTER SYSTEM SET
 effective_cache_size = '75GB';
ALTER SYSTEM SET
 maintenance_work_mem = '2GB';
ALTER SYSTEM SET
 checkpoint_completion_target = '0.9';
ALTER SYSTEM SET
 wal_buffers = '16MB';
ALTER SYSTEM SET
 default_statistics_target = '100';
ALTER SYSTEM SET
 random_page_cost = '1.1';
ALTER SYSTEM SET
 effective_io_concurrency = '200';
ALTER SYSTEM SET
 work_mem = '640MB';
ALTER SYSTEM SET
 min_wal_size = '1GB';
ALTER SYSTEM SET
 max_wal_size = '4GB';



-- -- Assume mem=200GB
-- ALTER SYSTEM SET shared_buffers = '50GB'; -- The value should be set to 15% to 25% of the machine’s total RAM.
-- ALTER SYSTEM SET work_mem = '1GB'; -- Total RAM * 0.25 / max_connections
-- ALTER SYSTEM SET maintenance_work_mem = '10GB'; -- Total RAM * 0.05
-- ALTER SYSTEM SET effective_cache_size = '100GB'; -- Recommendations are to set Effective_cache_size at 50% of the machine’s total RAM.
-- ALTER SYSTEM SET temp_buffers = '1.5GB';
-- -- ALTER SYSTEM RESET ALL;
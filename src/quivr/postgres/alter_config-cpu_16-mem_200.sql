-- DB Version: 12
-- OS Type: linux
-- DB Type: web
-- Total Memory (RAM): 200 GB
-- CPUs num: 16
-- Connections num: 100
-- Data Storage: ssd

ALTER SYSTEM SET
 max_connections = '100';
ALTER SYSTEM SET
 shared_buffers = '50GB';
ALTER SYSTEM SET
 effective_cache_size = '150GB';
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
 work_mem = '500MB'; -- Total RAM * 0.25 / max_connections
ALTER SYSTEM SET
 min_wal_size = '1GB';
ALTER SYSTEM SET
 max_wal_size = '4GB';
ALTER SYSTEM SET
 max_worker_processes = '16';
ALTER SYSTEM SET
 max_parallel_workers_per_gather = '4';
ALTER SYSTEM SET
 max_parallel_workers = '16';
ALTER SYSTEM SET
 max_parallel_maintenance_workers = '4';

# configure jobs to copy
import inspect
import os

from csrank.experiments.dbconnection_modified import ModifiedDBConnector

job_name = "tsp_experiment_1"
config_file_name = "db.json"
JOBS_TO_COPY = [96, 90, 86, 85]

# configure postgres database connector
DIR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
config_file_path = os.path.join(DIR_PATH, 'database_configs', config_file_name)
db_connector = ModifiedDBConnector(config_file_path=config_file_path, table_jobs="jobs_" + job_name)

# get jobs from db and one by one insert newly
db_connector.copy_configs(JOBS_TO_COPY)
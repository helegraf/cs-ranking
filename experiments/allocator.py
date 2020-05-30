"""Thin experiment runner which takes all simulation parameters from a database.

Usage:
  experiment_cv.py --config_file_name=<config_file_name> --job_name=<job_name>
  experiment_cv.py (-h | --help)

Arguments:
  FILE                  An argument for passing in a file.

Options:
  -h --help                             Show this screen.
  --config_file_name=<config_file_name>   File name of the database config
  --job_name=<job_name>             table from which to get configs
"""
import inspect
import json
import os
import subprocess
from pathlib import Path

from docopt import docopt

from csrank.experiments.dbconnection_modified import ModifiedDBConnector

arguments = docopt(__doc__)
config_file_name = arguments["--config_file_name"]
job_name = arguments["--job_name"]

# configure postgres database connector
DIR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), job_name, "experiments")
config_file_path = os.path.join(DIR_PATH, 'database_configs', config_file_name)
db_connector = ModifiedDBConnector(config_file_path=config_file_path, table_jobs="jobs_" + job_name)

# get all jobs that aren't finished from the table
jobs = db_connector.get_ready_jobs()

# create an allocation script for each job and execute them all
script = "#!/bin/bash\n" \
         "#CCS -N {}\n" \
         "#CCS --res=rset=1{}\n" \
         "#CCS -M hgraf@mail.upb.de\n" \
         "#CCS -ma\n" \
         "#CCS -t {}\n" \
         "\n" \
         "cd /upb/scratch/departments/pc2/groups/hpc-prf-isys/hgraf/attention\n" \
         "\n" \
         "# do the job\n" \
         "job_starter.sh -e {} -i {} -c ${{CCS_REQID}}\n"

# create folders
Path(os.path.join(DIR_PATH, "jobs", "out")).mkdir(parents=True, exist_ok=True)

# allocate jobs
for job in jobs:
    # assign variables
    job_identifier = job_name + "_" + str(job["job_id"])
    resources = ""
    resource_set = job["resources"]
    for resource_identifier in resource_set.keys():
        resources += ":" + resource_identifier + "=" + resource_set[resource_identifier]
    duration = job["duration"]

    # create script
    current_script = script.format(
        job_identifier,
        resources,
        duration,
        job_name,
        str(job["job_id"])
    )

    # write script to file
    path = os.path.join(DIR_PATH, "jobs", "job_{}.sh".format(str(job["job_id"])))
    script_file = open(path, "w")
    script_file.write(current_script)
    script_file.close()

    # cd
    target = "/upb/scratch/departments/pc2/groups/hpc-prf-isys/hgraf/attention/{}/experiments/logs/out\n"\
        .format(job_name)

    # echo allocation
    subprocess.run(["echo", "$PWD"])

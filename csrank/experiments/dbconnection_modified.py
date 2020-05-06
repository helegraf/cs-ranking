import hashlib
import json
import logging
import os
from abc import ABCMeta
from datetime import datetime

import psycopg2
from psycopg2.extras import DictCursor

from csrank.util import print_dictionary


class ModifiedDBConnector(metaclass=ABCMeta):

    def __init__(self, config_file_path, table_jobs, **kwargs):
        self.table_jobs = table_jobs
        self.logger = logging.getLogger('DBConnector')
        self.schema = "thesis_attention"
        self.job_description = None
        self.connection = None
        self.cursor_db = None
        if os.path.isfile(config_file_path):
            config_file = open(config_file_path, "r")
            config = config_file.read().replace('\n', '')
            self.logger.info("Config {}".format(config))
            self.connect_params = json.loads(config)
            self.logger.info("Connection Successful")
        else:
            raise ValueError('File does not exist for the configuration of the database')

    def init_connection(self, cursor_factory=DictCursor):
        self.connection = psycopg2.connect(**self.connect_params)
        if cursor_factory is None:
            self.cursor_db = self.connection.cursor()
        else:
            self.cursor_db = self.connection.cursor(cursor_factory=cursor_factory)

    def close_connection(self):
        self.connection.commit()
        self.connection.close()

    def get_job_for_id(self, cluster_id, job_id):
        # get job from db
        self.init_connection()
        jobs = "{}.{}".format(self.schema, self.table_jobs)
        select_job = """SELECT * FROM {0}  WHERE {0}.job_id={1}""".format(jobs, job_id)
        self.cursor_db.execute(select_job)

        if self.cursor_db.rowcount == 1:
            try:
                # update start time and set cluster job id
                self.job_description = self.cursor_db.fetchall()[0]
                print('Jobs found {}'.format(print_dictionary(self.job_description)))
                start = datetime.now()
                update_job = """UPDATE {} set time_start = %s, cluster_id = %s WHERE job_id = %s""".format(jobs)
                self.cursor_db.execute(update_job, (start, cluster_id, job_id))
                self.close_connection()

            except psycopg2.IntegrityError as e:
                print("IntegrityError for the job {}, already assigned to another node error {}".format(job_id, str(e)))
                self.job_description = None
                self.connection.rollback()

            except (ValueError, IndexError) as e:
                print("Error as the all jobs are already assigned to another nodes {}".format(str(e)))

        return self.job_description

    def insert_generalization_results(self, gen_results, gen_results_table_name):
        # TODO implement
        # gen_results_table_name should be the same as results_table_name but with "gen" as prefix
        # the generalization_results_table additionally needs fields num_test_objects and num_test_instances
        pass

    def insert_results(self, results, results_table_name, **kwargs):
        self.init_connection(cursor_factory=None)
        results_table = "{}.{}".format(self.schema, results_table_name)
        columns = ', '.join(list(results.keys()))
        values_str = ', '.join(list(results.values()))

        self.cursor_db.execute("select to_regclass(%s)", [results_table])
        is_table_exist = bool(self.cursor_db.fetchone()[0])
        if not is_table_exist:
            self.logger.info("Table {} does not exist creating with columns {}".format(results_table, columns))
            create_command = "CREATE TABLE {} (job_id INTEGER, train_test character varying(200))".format(
                results_table)
            self.cursor_db.execute(create_command)
            alter_command = "ALTER TABLE {} ADD PRIMARY KEY (job_id, train_test);".format(results_table)
            self.cursor_db.execute(alter_command)
            for column in results.keys():
                if column not in ["job_id", "train_test"]:
                    alter_table_command = 'ALTER TABLE %s ADD COLUMN %s double precision' % (results_table, column)
                    self.cursor_db.execute(alter_table_command)
            self.cursor_db.execute('ALTER TABLE %s ADD COLUMN time_updated timestamp' % results_table)
            self.cursor_db.execute('CREATE TRIGGER set_timestamp'
                                   'BEFORE UPDATE ON thesis_attention.{}'
                                   'FOR EACH ROW'
                                   'EXECUTE PROCEDURE trigger_set_timestamp();'.format(results_table))
            self.close_connection()
            self.init_connection(cursor_factory=None)

        try:
            insert_result = "INSERT INTO {0} ({1}) VALUES ({2})".format(results_table, columns, values_str)
            self.logger.info("Inserting results: {}".format(insert_result))
            self.cursor_db.execute(insert_result)
            if self.cursor_db.rowcount == 1:
                self.logger.info("Results inserted for the job {}".format(results['job_id']))
        except psycopg2.IntegrityError as e:
            self.logger.info(print_dictionary(results))
            self.logger.info(
                "IntegrityError for the job {0}, results already inserted to another node error {1}".format(
                    results["job_id"], str(e)))
            self.connection.rollback()
            update_str = ''
            values_tuples = []
            for i, col in enumerate(results.keys()):
                if col != 'job_id':
                    if (i + 1) == len(results):
                        update_str = update_str + col + " = %s "
                    else:
                        update_str = update_str + col + " = %s, "
                    if 'Infinity' in results[col]:
                        results[col] = 'Infinity'
                    values_tuples.append(results[col])
            update_result = "UPDATE {0} set {1} where job_id= %s ".format(results_table, update_str)
            self.logger.info(update_result)
            values_tuples.append(results['job_id'])
            self.logger.info('values {}'.format(tuple(values_tuples)))
            self.cursor_db.execute(update_result, tuple(values_tuples))
            if self.cursor_db.rowcount == 1:
                self.logger.info("The job {} is updated".format(results['job_id']))
        self.close_connection()

    def append_error_string_in_running_job(self, job_id, error_message):
        # get current error message
        self.init_connection(cursor_factory=None)
        jobs = "{}.{}".format(self.schema, self.table_jobs)
        current_message = "SELECT cluster_id, error_history from {0} WHERE {0}.job_id = {1}".format(jobs,
                                                                                                    job_id)
        # update message
        self.cursor_db.execute(current_message)
        cur_message = self.cursor_db.fetchone()
        if cur_message[1] is not None:
            error_message = error_message + ';\n' + cur_message[1]
        update_job = "UPDATE {0} SET error_history = %s WHERE job_id = %s".format(
            jobs)
        self.cursor_db.execute(update_job, (error_message, job_id))
        if self.cursor_db.rowcount == 1:
            self.logger.info("The job {} is interrupted".format(job_id))
        self.close_connection()

    def get_hash_value_for_job(self, job):
        keys = ['resources', 'dataset', 'dataset_params', 'fold_id', 'n_inner_folds', 'learning_problem', 'seed',
                'learner_name', 'learner_params', 'learner_fit_params', 'use_hp', 'hp_iterations', 'hp_ranges',
                'hp_fit_params', 'duration', 'time_out_eval', 'results_table_name']
        hash_string = ""
        for k in keys:
            hash_string = hash_string + str(k) + ':' + str(job[k])
        hash_object = hashlib.sha1(hash_string.encode())
        hex_dig = hash_object.hexdigest()
        self.logger.info("Job_id {} Hash_string {}".format(job.get('job_id', None), str(hex_dig)))
        return str(hex_dig)

    def insert_new_job(self, job):
        self.logger.info('Inserting job into db:')
        self.logger.info(print_dictionary(job))

        job.update({"hash_value": self.get_hash_value_for_job(job)})

        columns = ', '.join(list(job.keys()))
        values_str = self.convert_job_to_str(job.values())
        self.init_connection()

        table = "{}.{}".format(self.schema, self.table_jobs)
        new_job = "INSERT INTO {0} ({1}) VALUES ({2}) RETURNING job_id".format(table, columns, values_str)
        self.cursor_db.execute(new_job)

        # new job!
        id_new = -1
        if self.cursor_db.rowcount == 1:
            id_new = self.cursor_db.fetchone()[0]
            self.logger.info("Results inserted for the job {}".format(id_new))

        self.close_connection()

        return id_new

    def convert_job_to_str(self, list):
        values_str = []
        for i, val in enumerate(list):
            if isinstance(val, dict):
                val = "\'" + json.dumps(val) + "\'"
            elif isinstance(val, str):
                val = "\'" + val + "\'"
            else:
                val = str(val)
            values_str.append(val)
            if i == 0:
                values = '%s'
            else:
                values = values + ', %s'
        values_str = values % tuple(values_str)
        return values_str

    def insert_validation_loss(self, validation_loss, job_id):
        self.logger.info("Inserting validation loss into db.")

        self.init_connection()
        jobs = "{}.{}".format(self.schema, self.table_jobs)
        update_job = """UPDATE {} set validation_loss = %s WHERE job_id = %s""".format(jobs)
        self.cursor_db.execute(update_job, (str(validation_loss), str(job_id)))
        self.close_connection()

    def finish_job(self, job_id, cluster_id):
        """

        Updates the jobs table row so that the time_finished is entered.

        Parameters
        ----------
        cluster_id : the cluster id of the job for which the time_finished should be inserted
        job_id : the id of the job for which the time_finished should be inserted
        """
        self.init_connection()
        jobs = "{}.{}".format(self.schema, self.table_jobs)
        update_job = """UPDATE {} set time_finished = %s WHERE job_id = %s""".format(jobs)
        curr_time = datetime.now()
        self.cursor_db.execute(update_job, (curr_time, str(job_id)))
        self.close_connection()

    def update_time_finished_train(self, job_id):
        self.init_connection()
        jobs = "{}.{}".format(self.schema, self.table_jobs)
        update_job = """UPDATE {} set time_finished_train = %s WHERE job_id = %s""".format(jobs)
        curr_time = datetime.now()
        self.cursor_db.execute(update_job, (curr_time, str(job_id)))
        self.close_connection()

    def get_ready_jobs(self):
        self.init_connection()
        jobs = "{}.{}".format(self.schema, self.table_jobs)

        self.cursor_db.execute("SELECT job_id, resources, duration FROM {} WHERE time_start IS NULL".format(jobs))
        records = self.cursor_db.fetchall()
        data = []
        for row in records:
            data.append({"job_id": row[0], "resources": row[1], "duration": row[2]})
        self.close_connection()

        return data

    def copy_configs(self, jobs_to_copy):
        self.init_connection()
        jobs = "{}.{}".format(self.schema, self.table_jobs)

        for job in jobs_to_copy:
            self.cursor_db.execute("SELECT resources, dataset, dataset_params, fold_id, n_inner_folds, "
                                   "learning_problem, seed, learner_name, learner_params, learner_fit_params, use_hp, "
                                   "hp_iterations, hp_ranges, hp_fit_params, duration, time_out_eval, "
                                   "results_table_name, hash_value "
                                   "FROM {} WHERE job_id={}".format(jobs, job))
            record = self.cursor_db.fetchone()
            print(record)

            record_str = self.convert_job_to_str(record)

            self.cursor_db.execute("INSERT INTO {} ("
                                   "resources, dataset, dataset_params, fold_id, n_inner_folds, learning_problem, "
                                   "seed, learner_name, learner_params, learner_fit_params, use_hp, hp_iterations, "
                                   "hp_ranges, hp_fit_params, duration, time_out_eval, results_table_name, hash_value"
                                   ") VALUES ({})".format(jobs, record_str))

        self.close_connection()

    def get_results_for_job(self, job_ids, results_table_name):
        self.init_connection()

        jobs = "{}.{}".format(self.schema, results_table_name)
        self.cursor_db.execute("SELECT * FROM {} WHERE job_id IN ({})".format(jobs, str(job_ids)[1:-1]))
        colnames = [desc[0] for desc in self.cursor_db.description]
        results = self.cursor_db.fetchall()

        self.close_connection()

        return colnames, results

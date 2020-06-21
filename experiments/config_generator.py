import copy
import json

tables = {
    1: {
        10: [[1],
             [2],
             [3],
             [4],
             [5],
             [6],
             [7],
             [8],
             [9],
             [10]]
    },
    2: {
        10: [[2, 9],
             [8, 10],
             [10, 6],
             [4, 7],
             [1, 3],
             [5, 1],
             [9, 2],
             [7, 4],
             [6, 8],
             [3, 5]]
    },
    4: {
        10: [[1, 4, 3, 5],
             [5, 6, 7, 1],
             [8, 2, 2, 2],
             [4, 7, 1, 9],
             [6, 3, 10, 6],
             [7, 10, 4, 7],
             [10, 8, 5, 4],
             [9, 5, 8, 10],
             [2, 9, 9, 3],
             [3, 1, 6, 8]],
        15: [[9, 1, 4, 10],
             [11, 14, 5, 5],
             [10, 7, 10, 1],
             [3, 12, 3, 2],
             [13, 10, 2, 13],
             [5, 4, 8, 15],
             [2, 2, 12, 6],
             [12, 5, 14, 11],
             [1, 8, 6, 12],
             [7, 6, 1, 7],
             [4, 15, 9, 9],
             [14, 3, 7, 3],
             [6, 9, 15, 4],
             [15, 11, 11, 8],
             [8, 13, 13, 14]]
    },
    3: {
        10: [[6, 7, 1],
             [2, 8, 8],
             [5, 3, 10],
             [1, 5, 4],
             [7, 1, 7],
             [9, 6, 9],
             [4, 10, 6],
             [10, 4, 5],
             [8, 9, 3],
             [3, 2, 2]]
    },
    5: {
        10: [[4, 6, 6, 10, 1],
             [6, 2, 8, 1, 4],
             [1, 1, 4, 7, 6],
             [3, 5, 2, 3, 9],
             [5, 10, 7, 6, 10],
             [10, 7, 5, 2, 7],
             [2, 8, 9, 4, 3],
             [7, 9, 1, 8, 5],
             [8, 4, 10, 9, 8],
             [9, 3, 3, 5, 2]]
    },
    6: {
        10: [[6, 7, 1, 7, 1, 7],
             [4, 1, 4, 5, 10, 8],
             [7, 2, 8, 1, 2, 5],
             [1, 5, 2, 2, 7, 4],
             [9, 3, 3, 9, 5, 2],
             [2, 4, 9, 8, 4, 9],
             [8, 6, 10, 6, 9, 3],
             [3, 9, 6, 4, 3, 1],
             [10, 8, 5, 3, 6, 10],
             [5, 10, 7, 10, 8, 6]]
    }
}


def get_configs_for(params_dict, num_runs, config_dict):
    # 1. look up table in tables
    num_params = len(params_dict)
    table = tables[num_params][num_runs]

    configurations = {}

    param_index = 0
    for param, values in params_dict.items():

        param_values = []

        for run in range(num_runs):
            num = table[run][param_index] / num_runs * len(values)
            # add epsilon so that num is rounded down
            num = max(num - 1e-10, 0)
            param_value_index = int(num)

            param_values.append(values[param_value_index])

        configurations[param] = param_values

        param_index += 1

    confs = []
    for run in range(num_runs):
        conf_dict = copy.deepcopy(config_dict)

        for param in params_dict.keys():
            replace_param_from_json(param, configurations[param][run], conf_dict)

        confs.append(conf_dict)

    return confs


def replace_param_from_json(param, value, conf_dict):
    for key, val in conf_dict.items():
        if key == param:
            conf_dict[key] = value
        elif isinstance(val, dict):
            replace_param_from_json(param, value, conf_dict[key])


def read_config_file(filename):
    configuration_file = open(filename, "r")
    config_str = configuration_file.read().replace('\n', '')
    configuration_file.close()
    return json.loads(config_str)


def write_json_to_file(filename, json_dict):
    print("writing to file", filename)
    configuration_file = open(filename, "w")
    configuration_file.write(json.dumps(json_dict))
    configuration_file.flush()
    configuration_file.close()


def read_json_and_replace(dataset, learner, num_runs):
    filename_learner = "experiment_configs/{}/base_configs/{}_{}_base.json".format(dataset, dataset, learner)
    json_conf_learner = read_config_file(filename_learner)
    config_dict = json_conf_learner["learners"][0]

    filename_params = "experiment_configs/{}/{}_params.json".format(dataset, dataset)
    json_conf_params = read_config_file(filename_params)
    params_dict = json_conf_params[learner]

    confs = get_configs_for(params_dict, num_runs, config_dict)
    final_config = json_conf_learner
    final_config['learners'] = confs
    write_json_to_file("experiment_configs/{}/generated_configs/{}_{}.json"
                       .format(dataset, dataset, learner), final_config)


dataset = 'simple_ranking'
learner = 'set_transformer_ranker'
num_confs = 10

read_json_and_replace(dataset=dataset,
                      learner=learner,
                      num_runs=num_confs)

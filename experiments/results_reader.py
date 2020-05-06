import h5py
import numpy as np
import matplotlib.pyplot as plt

from csrank.visualization.predictions import tsp_figure


def check_predictions_file():
    prediction_file = "predictions/test.h5"
    f = h5py.File(prediction_file, 'r')

    print(f)
    print(f.values())
    print(f.keys())
    print(f.get('scores'))

    print(np.array(f.get('scores')))

    f.close()


def check_visualizations_file():
    visualizations_file = 'tensorboard_logs/tsp_experiment_3/fate_ranker/save_vis_test/visualizations.h5'
    f = h5py.File(visualizations_file, 'r')

    print(f)
    print(f.values())
    print(f.keys())

    figure = tsp_figure(f.get('inputs')[0], f.get('targets')[0], f.get('outputs')[0], f.get('metric')[0], f.get('epoch')[0])
    plt.show()


check_visualizations_file()
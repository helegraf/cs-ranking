import os

import h5py
import numpy as np
from pygmo import hypervolume
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.datasets import make_regression
from sklearn.datasets import make_blobs
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import check_random_state
import elkai

from csrank.constants import OBJECT_RANKING
from csrank.numpy_util import scores_to_rankings
from ..synthetic_dataset_generator import SyntheticDatasetGenerator
from ..util import create_pairwise_prob_matrix, quicksort
from ...util import create_dir_recursively


def load_tsp_dataset(n_train_instances, n_test_instances, n_objects, seed, path):
    filename = "tsp_dataset_n_train_instances-{}_n_test_instances-{}_n_objects-{}_seed-{}"
    filename = filename.format(n_train_instances, n_test_instances, n_objects, seed)
    pred_file = os.path.join(path, filename + ".h5")
    f = h5py.File(pred_file, 'r')

    x_train = np.array(f.get('x_train'))
    x_test = np.array(f.get('x_test'))
    y_train = np.array(f.get('y_train'))
    y_test = np.array(f.get('y_test'))

    f.close()

    return x_train, y_train, x_test, y_test


def save_tsp_dataset(x_train, y_train, x_test, y_test, n_train_instances, n_test_instances, n_objects, seed,
                     path):
    filename = "tsp_dataset_n_train_instances-{}_n_test_instances-{}_n_objects-{}_seed-{}"
    filename = filename.format(n_train_instances, n_test_instances, n_objects, seed)
    pred_file = os.path.join(path, filename + ".h5")
    create_dir_recursively(pred_file, True)
    f = h5py.File(pred_file, 'w')

    f.create_dataset('x_train', data=x_train)
    f.create_dataset('x_test', data=x_test)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('y_test', data=y_test)

    f.close()


class ObjectRankingDatasetGenerator(SyntheticDatasetGenerator):
    def __init__(self, dataset_type='medoid', **kwargs):
        super(ObjectRankingDatasetGenerator, self).__init__(
            learning_problem=OBJECT_RANKING, **kwargs)
        dataset_function_options = {'linear': self.make_linear_transitive,
                                    'medoid': self.make_intransitive_medoids,
                                    'gp_transitive': self.make_gp_transitive,
                                    'gp_non_transitive': self.make_gp_non_transitive,
                                    "hyper_volume": self.make_hv_dataset,
                                    'tsp': self.make_tsp_dataset,
                                    'simple_max': self.make_simple_max_choice,
                                    'min_max': self.make_min_max_choice}
        if dataset_type not in dataset_function_options.keys():
            dataset_type = "medoid"
        self.dataset_function = dataset_function_options[dataset_type]

    def make_min_max_choice(self, n_instances, n_objects, n_features, seed=42, **kwargs):
        # generate objects with a weight and value
        random_state = check_random_state(seed)
        x = random_state.uniform(low=0, high=1, size=(n_instances, n_objects, n_features))

        # use max to determine y value
        y = np.empty((n_instances, n_objects), dtype=int)
        for instance in range(n_instances):
            # select the lowest arg for all objects
            minima = np.min(x[instance], axis=1)
            y[instance] = np.asarray(np.flip(np.argsort(minima)), dtype=int)

        return x, y

    def make_simple_max_choice(self, n_instances, n_objects, seed=42, **kwargs):
        print("instances, objects", n_instances, n_objects)
        # generate objects with a weight and value
        random_state = check_random_state(seed)
        x = random_state.uniform(low=0, high=1, size=(n_instances, n_objects))

        # use max to determine y value
        y = np.empty((n_instances, n_objects), dtype=int)
        for instance in range(n_instances):
            y[instance] = np.asarray(np.flip(np.argsort(x[instance])), dtype=int)

        x = x.reshape(n_instances, n_objects, 1)

        return x, y

    def make_tsp_dataset(self, n_instances, n_objects, seed=42, **kwargs):
        # 1. Generate x data
        random_state = check_random_state(seed)
        x = random_state.random_integers(low=0, high=10000, size=(n_instances, n_objects, 2))

        # 2. Label the data
        y = np.empty((n_instances, n_objects), dtype=int)
        for instance in range(n_instances):
            # fill matrix with distances
            matrix = cdist(x[instance], x[instance])

            # convert distances to int
            matrix = matrix.astype(int)

            # predict labels
            y[instance] = np.asarray(np.argsort(elkai.solve_int_matrix(matrix)), dtype=int)

        return x, y

    def make_linear_transitive(self, n_instances=1000, n_objects=5, noise=0.0, n_features=100, n_informative=10,
                               seed=42, **kwd):
        random_state = check_random_state(seed=seed)
        X, y, coeff = make_regression(n_samples=n_instances * n_objects,
                                      n_features=n_features,
                                      n_informative=n_informative, coef=True,
                                      noise=noise,
                                      random_state=random_state)
        X = X.reshape(n_instances, n_objects, n_features)
        y = y.reshape(n_instances, n_objects)
        Y = scores_to_rankings(y)
        return X, Y

    def make_gp_transitive(self, n_instances=1000, n_objects=5, noise=0.0, n_features=100, kernel_params=None, seed=42,
                           **kwd):
        """Creates a nonlinear object ranking problem by sampling from a
        Gaussian process as the latent utility function.
        Note that this function needs to compute a kernel matrix of size
        (n_instances * n_objects) ** 2, which could allocate a large chunk of the
        memory."""
        random_state = check_random_state(seed=seed)

        if kernel_params is None:
            kernel_params = dict()
        n_total = n_instances * n_objects
        X = random_state.rand(n_total, n_features)
        L = np.linalg.cholesky(Matern(**kernel_params)(X))
        f = (L.dot(random_state.randn(n_total)) +
             random_state.normal(scale=noise, size=n_total))
        X = X.reshape(n_instances, n_objects, n_features)
        f = f.reshape(n_instances, n_objects)
        Y = scores_to_rankings(f)

        return X, Y

    def make_gp_non_transitive(self, n_instances=1000, n_objects=5, n_features=100, center_box=(-10.0, 10.0),
                               cluster_std=2.0, seed=42, **kwd):
        n_samples = n_instances * n_objects
        random_state = check_random_state(seed=seed)
        x, y = make_blobs(n_samples=n_samples, centers=n_objects, n_features=n_features, cluster_std=cluster_std,
                          center_box=center_box, random_state=random_state, shuffle=True)
        y = np.array([y])
        samples = np.append(x, y.T, axis=1)
        samples = samples[samples[:, n_features].argsort()]
        pairwise_prob = create_pairwise_prob_matrix(n_objects)
        X = []
        Y = []
        for inst in range(n_instances):
            feature = np.array([samples[inst + i * n_instances, 0:-1] for i in
                                range(n_objects)])
            matrix = np.random.binomial(1, pairwise_prob)
            objects = list(np.arange(n_objects))
            ordering = np.array(quicksort(objects, matrix))
            ranking = np.argsort(ordering)
            X.append(feature)
            Y.append(ranking)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def make_intransitive_medoids(self, n_instances=100, n_objects=5, n_features=100, seed=42, **kwd):
        random_state = check_random_state(seed=seed)
        X = random_state.uniform(size=(n_instances, n_objects, n_features))
        Y = np.empty((n_instances, n_objects))
        for i in range(n_instances):
            D = squareform(pdist(X[i], metric='euclidean'))
            sum_dist = D.mean(axis=0)
            medoid = np.argmin(sum_dist)
            ordering = np.argsort(D[medoid])
            ranking = np.argsort(ordering)
            Y[i] = ranking
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def make_hv_dataset(self, n_instances=1000, n_objects=5, n_features=5, seed=42, **kwd):
        random_state = check_random_state(seed=seed)
        X = random_state.randn(n_instances, n_objects, n_features)
        # Normalize to unit circle and fold to lower quadrant
        X = -np.abs(X / np.sqrt(np.power(X, 2).sum(axis=2))[..., None])
        Y = np.empty((n_instances, n_objects), dtype=int)
        reference = np.zeros(n_features)
        for i, x in enumerate(X):
            hv = hypervolume(x)
            cont = hv.contributions(reference)
            Y[i] = np.argsort(cont)[::-1].argsort()

        return X, Y

    def get_single_train_test_split(self):
        return super(ObjectRankingDatasetGenerator, self).get_single_train_test_split()

    def get_train_test_datasets(self, n_datasets=5):
        return super(ObjectRankingDatasetGenerator, self).get_train_test_datasets(n_datasets=n_datasets)

    def get_dataset_dictionaries(self, lengths=[5, 6]):
        return super(ObjectRankingDatasetGenerator, self).get_dataset_dictionaries(lengths=lengths)

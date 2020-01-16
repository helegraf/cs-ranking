from csrank.constants import OBJECT_RANKING
from csrank.dataset_reader.dataset_reader import DatasetReader
import numpy as np
from sklearn.utils import check_random_state


class TSPDatasetReader(DatasetReader):
    def __init__(self, n_features=2, n_objects=5, n_train_instances=40, n_test_instances=10, random_state=None, **kwargs):
        """
        A function that does something and is the same

        Parameters
        ----------
        n_features : The number of features
        n_objects : The number of objects
        n_train_instances :
        n_test_instances :
        random_state :
        kwargs : hello

        Returns
        -------
        Something as well
        """

        super(TSPDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, dataset_folder="data", **kwargs)

        self.nFeatures = n_features
        self.nObjects = n_objects
        self.nTrainInstances = n_train_instances
        self.nTestInstances = n_test_instances
        self.random_state = check_random_state(random_state)

        self.__load_dataset__()
        
    def __load_dataset__(self):
        """
        Does something

        Returns
        -------
        self.nFeatures : int
            an int
        string : str
            a string
        """

        print("hello i am loadinggg")
        self.create_dataset(datatype='train')
        self.create_dataset(datatype='test')

        return self.nFeatures, "abc"

    def splitter(self, iter_, other):
        """
        Splits

        :param iter_:
        :param other:
        """
        for i in iter_:
            self.X, self.Y = X_train, Y_train = self.make_dataset(datatype='train', seed=10 * i + 32)
            self.__check_dataset_validity__()

            self.X, self.Y = X_test, Y_test = self.make_dataset(datatype='test', seed=10 * i + 32)
            self.__check_dataset_validity__()

        yield X_train, Y_train, X_test, Y_test

    def get_dataset_dictionaries(self):
        """
        Short description

        long description [1]_

        Returns
        -------
        8

        Notes
        -------
        .. math::
            1 \leq 3

        References
        -------
        .. [1] a ref

        """
        return 8

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.X, self.Y = X_train, Y_train = self.make_dataset(datatype='train', seed=seed)
        self.__check_dataset_validity__()

        self.X, self.Y = X_test, Y_test = self.make_dataset(datatype='test', seed=seed + 1)
        self.__check_dataset_validity__()
        return X_train, Y_train, X_test, Y_test

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        return self.splitter(splits)

    def create_dataset(self, datatype):
        pass

    def make_dataset(self, datatype, seed):
        if datatype == "train":
            n_instances = self.nTrainInstances
        else:
            n_instances = self.nTestInstances
        train = np.empty((n_instances, self.nObjects, self.nFeatures), dtype=float)
        test = train  # for now
        return train, test


reader = TSPDatasetReader()
reader.get_single_train_test_split()
import h5py
import numpy as np

files = ['c85ca80772e6c997f9c0c2877c45bd4e73ff3fa3', 'c957168ea787121897430b134e693dfbd7fa3a00',
         '3232e612f474c4ae28460ac53bdde9515722f5f0', '57e67eeba91a5e626d1c02031975813db6b59fef',
         '292383293104f61f06f43a02c283a0ea1a42f4b0', '9de91d424f273fc58f69a84e34d884f6537f47be',
         '6c54ec395f9325657e139cdd98aa315cb2891016', '4bc4f0e8be4a839b2555bbc472d650ddc5417964',
         '56813fda18e2e771709c2db8bab1f19ab370287b', '74b7f34182722899f8dc37ab0f7d2036c63c18f6',
         'b150060158dfbe2dedecddcfb8cce77f36ceb180']

seed = 1234
fold_id = 1

# generate data
random_state = np.random.RandomState(seed=seed + fold_id)
dataset_params['random_state'] = random_state
dataset_params['fold_id'] = fold_id
dataset_reader = get_dataset_reader(dataset_name, dataset_params)
x_train, y_train, x_test, y_test = dataset_reader.get_single_train_test_split()

for file in files:
    f = h5py.File('predictions/tsp_experiment_2/{}_train.h5'.format(file), 'r')
    print(list(f.keys()))
    dset = f['scores']
    print(dset.shape)
    print(dset.dtype)
    print(dset)

    arr = dset[()]
    print(file + "_train")
    print("Any nan?", np.isnan(arr).any())
    print('Unique values:', np.unique(arr))

    f = h5py.File('predictions/tsp_experiment_2/{}_test.h5'.format(file), 'r')
    print(list(f.keys()))
    dset = f['scores']
    print(dset.shape)
    print(dset.dtype)
    print(dset)

    arr = dset[()]
    print(file + "_test")
    print("Any nan?", np.isnan(arr).any())
    print('Unique values:', np.unique(arr))
from keras import backend as K


def combine_batch_to_features(batch_size, feature_dimension_after, feature_dimension_previous, num_heads, num_objects,
                              matrix):
    # [h * b, n, p] -> [h, b, n, p]
    split_batch = K.reshape(matrix, shape=(num_heads, batch_size, num_objects, feature_dimension_after))
    # [h, b, n, p] -> [b, n, h, p]
    swap_dimensions_back = K.permute_dimensions(split_batch, pattern=(1, 2, 0, 3))
    # [b, n, h, p] -> [b, n, d]
    combine_features = K.reshape(swap_dimensions_back, shape=(batch_size, num_objects, feature_dimension_previous))
    return combine_features


def split_features_to_batch(batch_size, feature_dimension_after, num_heads, num_objects, matrix):
    # [b, n, d] -> [b, n, h, p]
    split_features = K.reshape(matrix, shape=(batch_size, num_objects, num_heads, feature_dimension_after))
    # [b, n, h, p] -> [h, b, n, p]
    swap_dimensions = K.permute_dimensions(split_features, pattern=(2, 0, 1, 3))
    # [h, b, n, p] -> [h * b, n, p]
    combine_batch = K.reshape(swap_dimensions, shape=(num_heads * batch_size, num_objects, feature_dimension_after))
    return combine_batch



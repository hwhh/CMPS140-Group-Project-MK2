import os
import pandas as pd
from pandas.compat import StringIO
import numpy as np
from keras import backend as K
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io.file_io import stat
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 models into TensorFlow SavedModel."""
    builder = saved_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def is_file_available(filepath):
    try:
        return stat(filepath)
    except NotFoundError as e:
        return False


def save_file(dir, file, name):
    if dir.startswith('gs://'):
        np.save(dir + name, file)
        copy_file_to_gcs(dir, 'features.npy')
    else:
        np.save(dir + name, file)


def load_file(job_dir, file_name):
    if job_dir.startswith('gs://') and is_file_available(job_dir + file_name):
        with file_io.FileIO(job_dir + file_name, mode='r') as input_f:
            return np.load(input_f), True
    elif os.path.exists(os.path.join(job_dir + file_name)):
        return np.load(job_dir + file_name), True
    else:
        return [], False


def create_predictions(config):
    file_stream = file_io.FileIO(config.train_csv[0], mode='r')
    train = pd.read_csv(StringIO(file_stream.read()))
    LABELS = list(train.label.unique())
    pred_list = []
    for i in range(config.n_folds):
        with file_io.FileIO(config.job_dir + '/test_predictions_%d.npy' % i, mode='r') as input_f:
            pred_list.append(np.load(input_f))

    prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        prediction = prediction * pred
    prediction = prediction ** (1. / len(pred_list))
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    file_stream = file_io.FileIO(config.test_csv[0], mode='r')
    test = pd.read_csv(StringIO(file_stream.read()))
    test['label'] = predicted_labels
    if config.job_dir.startswith("gs://"):
        test[['fname', 'label']].to_csv('1d_2d_ensembled_submission.csv', index=False)
        copy_file_to_gcs(config.job_dir, '1d_2d_ensembled_submission.csv')
    else:
        test[['fname', 'label']].to_csv(os.path.join(config.job_dir, '1d_2d_ensembled_submission.csv'), index=False)
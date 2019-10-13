from .imports import *

import urllib.request
from tqdm import tqdm_notebook

# Get recommended directories for storing datasets (data_dir) and intermediate files generated during training
# (work_dir).
# :param root_dir: Root directory, which is often the Google Cloud Storage bucket when using TPUs.
# :param project: Name of the project.
# :return: Data directory for storaing datasets, and work directory for storing intermediate files.   
def get_project_dirs(root_dir: str, project: str) -> Tuple[str, str]:
    data_dir: str = os.path.join(root_dir, 'data', project)
    work_dir: str = os.path.join(root_dir, 'work', project)
    gfile.makedirs(data_dir)
    gfile.makedirs(work_dir)
    return data_dir, work_dir

def float_tffeature(value) -> tf.train.Feature:
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int_tffeature(value) -> tf.train.Feature:
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_tffeature(value) -> tf.train.Feature:
    if isinstance(value, str):
        value = bytes(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def numpy_tfexample(x, y=None) -> tf.train.Example:
    if y is None:
        feat_dict = {'image': float_tffeature(x.tolist())}
    else:
        feat_dict = {'image': float_tffeature(x.tolist()), 'label': int_tffeature(y)}
    return tf.train.Example(features=tf.train.Features(feature=feat_dict))

def numpy_tfrecord(output_fn: str, X, y=None, overwrite: bool = False):
    n = X.shape[0]
    X_reshape = X.reshape(n, -1)

    if overwrite or not tf.io.gfile.exists(output_fn):
        with tf.io.TFRecordWriter(output_fn) as record_writer:
            for i in tqdm_notebook(range(n)):
                example = numpy_tfexample(X_reshape[i]) if y is None else numpy_tfexample(X_reshape[i], y[i])
                record_writer.write(example.SerializeToString())
    else:
        tf.logging.info('Output file already exists. Skipping.')

def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())
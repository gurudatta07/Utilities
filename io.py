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

def tfexample_numpy_image_parser(tfexample: tf.train.Example, h: int, w: int, c: int = 3, dtype=tf.float32) -> Tuple[
    tf.Tensor, tf.Tensor]:
  """
    Parse a given TFExample containing an (image, label) pair, where the image is represented as an 3D array of shape
    [h*w*c] (i.e., flattened).
    :param tfexample: An input TFExample.
    :param h: Height of the image.
    :param w: Weight of the image.
    :param c: Number of color channels. Default to 3 (RGB).
    :param dtype: Data type of the returned image.
    :return: Parsed image and label Tensors.
  """
  feat_dict = {'image': tf.FixedLenFeature([h * w * c], dtype),
               'label': tf.FixedLenFeature([], tf.int64)}
  feat = tf.parse_single_example(tfexample, features=feat_dict)
  x, y = feat['image'], feat['label']
  x = tf.reshape(x, [h, w, c])
  return x, y

def tfrecord_fetch_dataset(fn: str) -> tf.data.Dataset:
  """
    Create a `tf.data` dataset from a given TFRecord file name.
    :param fn: Name of the TFRecord file.
    :return: `tf.data` dataset.
  """
  buffer_size = 8 * 1024 * 1024  # 8 MiB per file
  dataset = tf.data.TFRecordDataset(fn, buffer_size=buffer_size)
  return dataset
  
def tfrecord_ds(file_pattern: str, parser, batch_size: int, training: bool = True, shuffle_buf_sz: int = 50000,
                n_cores: int = 2, n_folds: int = 1, val_fold_idx: int = 0, streaming: bool = False) -> tf.data.Dataset:

  """
    Create a `tf.data` input pipeline from TFRecords files whose names satisfying a given pattern. Optionally partitions
    the data into training and validation sets according to k-fold cross-validation requirements.

    :param file_pattern: file pattern such as `data_train*.tfrec`
    :param parser: TFRecords parser function, which may also perform data augmentations.
    :param batch_size: Size of a data batch.
    :param training: Whether this is a training dataset, in which case the dataset is randomly shuffled and repeated.
    :param shuffle_buf_sz: Shuffle buffer size, for shuffling a training dataset. Default: 50k records.
    :param n_cores: Number of CPU cores, i.e., parallel threads.
    :param n_folds: Number of cross validation folds. Default: 1, meaning no cross validation.
    :param val_fold_idx: Fold ID for validation set, in cross validation. Ignored when `n_folds` is 1.
    :param streaming: under construction.
    :return: a `tf.data` dataset satisfying the above descriptions.
  """
  if streaming:
      # fixme
      dataset = tpu_datasets.StreamingFilesDataset(file_pattern, filetype='tfrecord', batch_transfer_size=batch_size)
  else:
      dataset = tf.data.Dataset.list_files(file_pattern)
      fetcher = tf.data.experimental.parallel_interleave(tfrecord_fetch_dataset, cycle_length=n_cores, sloppy=True)
      dataset = dataset.apply(fetcher)

  mapper_batcher = tf.data.experimental.map_and_batch(parser, batch_size=batch_size, num_parallel_batches=n_cores,
                                                      drop_remainder=True)

  if n_folds > 1:
      dataset = crossval_ds(dataset, n_folds, val_fold_idx, training)

  if training:
      dataset = dataset.shuffle(shuffle_buf_sz)
      dataset = dataset.repeat()

  dataset = dataset.apply(mapper_batcher)
  dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
  return dataset


def create_clean_dir(path: str):
  """
    Create a new directory specified by `path`. If this directory already exists, delete all its files and
    subdirectories.
    :param path: Path to the directory to be created or cleaned.
    :return: None
  """
  if gfile.exists(path):
    gfile.rmtree(path)
  gfile.makedirs(path)
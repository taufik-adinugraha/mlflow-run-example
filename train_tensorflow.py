import mlflow.tensorflow
import mlflow.keras
import mlflow
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import warnings

# data preprocessing
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

if __main__ == 'main':
    warnings.filterwarnings("ignore")

    # connect to tracking URI
    # URI can either be a HTTP/HTTPS URI for a remote server, or a local path to log data to a directory
    mlflow.set_tracking_uri('./myml') 

    # set experiment name to organize runs
    experiment_name = 'MNIST'
    mlflow.set_experiment(experiment_name)

    # automatic logging allows us to log metrics, parameters, and models without the need for explicit log statements
    mlflow.keras.autolog() 

    # parameters
    # epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    # learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    epochs = int(sys.argv[1])
    learning_rate = float(sys.argv[2])

    # import some data to play with
    (ds_train, ds_val), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(128)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)


    # build model
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # train
    with mlflow.start_run() as run:
        model.fit(ds_train, epochs=epochs, validation_data=ds_val)
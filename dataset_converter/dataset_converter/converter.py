import tensorflow as tf
import numpy as np
import os


def save_tf_dataset(loader, dataset_name: str, output_path: str | None = None):
    output_path = "./datasets/" if not output_path else output_path

    save_dir = os.path.join(output_path, dataset_name)
    data_batches = []
    labels_batches = []

    for data, labels in loader:
        data_batches.append(data.numpy())
        labels_batches.append(labels.numpy())

    data = tf.convert_to_tensor(np.concatenate(data_batches, axis=0))
    labels = tf.convert_to_tensor(
        np.concatenate(labels_batches, axis=0), dtype=np.uint32
    )
    print(data.shape)
    print(labels)
    tf_data = tf.data.Dataset.from_tensor_slices((data, labels))
    print(tf_data)
    print("saving dataset at:", save_dir)
    tf_data.save(save_dir, compression="GZIP")
    print("saved)")

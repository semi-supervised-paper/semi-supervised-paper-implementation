import numpy as np
import os
import pickle
import scipy.misc

data_dir = 'data_raw'  # path of CIFAR-10 data_batch_{1, 2, 3, 4, 5}
num_labels = 4000  # how many training data is labeled
target_dir = '../data_local/images/cifar/cifar10/by-image'  # path used to save CIFAR-10 images
file_dir = '../data_local/labels/cifar10'  # folder used to save path of unlabeled data


def load_file(file_name):
    with open(os.path.join(data_dir, file_name), 'rb') as meta_f:
        return pickle.load(meta_f, encoding="latin1")


def save_img(img, name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, name)
    if os.path.exists(filepath):
        return
    if len(img.shape) == 3:
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = np.transpose(img, (1, 2, 0))

    scipy.misc.toimage(img).save(filepath)


label_names = load_file('batches.meta')['label_names']
print("Found {} label names: {}".format(len(label_names), ", ".join(label_names)))


def load_cifar_10(unpack=False):

    def load_cifar_batches(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []
        for fn in filenames:
            with open(os.path.join(data_dir, fn), 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            images.append(np.asarray(data['data'], dtype='int32').reshape(-1, 3, 32, 32))
            labels.append(np.asarray(data['labels'], dtype='int32'))

        return np.concatenate(images), np.concatenate(labels)

    X_train, y_train = load_cifar_batches(['data_batch_%d' % i for i in (1, 2, 3, 4, 5)])
    X_test, y_test = load_cifar_batches('test_batch')

    if unpack:
        for i in range(X_train.shape[0]):
            name = "{}_{}.png".format(i, label_names[y_train[i]])
            save_img(X_train[i], name, os.path.join(target_dir, "train+val", label_names[y_train[i]]))
        for i in range(X_test.shape[0]):
            name = "{}_{}.png".format(i, label_names[y_test[i]])
            save_img(X_test[i], name, os.path.join(target_dir, "test", label_names[y_test[i]]))

    return X_train, y_train, X_test, y_test


def prepare_dataset(X_train, y_train, X_test, y_test, num_classes):
    train_file_name_list = []
    for i in range(X_train.shape[0]):
        name = "{}_{}.png".format(i, label_names[y_train[i]])
        train_file_name_list.append(name)
    train_file_name = np.array(train_file_name_list)

    # Random shuffle.

    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    train_file_name = train_file_name[indices]

    # Reshuffle.
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    train_file_name = train_file_name[indices]
    # Construct mask_train. It has a zero where label is unknown, and one where label is known.
    if num_labels == 'all':
        # All labels are used.
        mask_train = np.ones(len(y_train), dtype=np.float32)
        print("Keeping all labels.")
    else:
        # Assign labels to a subset of inputs.
        max_count = num_labels // num_classes
        print("Keeping %d labels per class." % max_count)
        mask_train = np.zeros(len(y_train), dtype=np.float32)
        count = [0] * num_classes
        for i in range(len(y_train)):
            label = y_train[i]
            if count[label] < max_count:
                mask_train[i] = 1.0
            count[label] += 1

    return X_train, y_train, mask_train, X_test, y_test, train_file_name


def produce_labeled_filename(X_train, y_train, mask_train, train_file_name, seed):
    folder = os.path.join(file_dir, "{}_balanced_labels".format(num_labels))
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "{}.txt".format(seed))
    if os.path.exists(file_path):
        print("file {} has existed")
        return
    file = open(file_path, 'w')
    counter = 0
    print(X_train.shape)
    for i in range(X_train.shape[0]):
        name = train_file_name[i]
        if mask_train[i] == 1.0:
            file.write(name + " " + label_names[y_train[i]] + "\n")
            counter += 1
    file.close()
    print("write to file {} labeled images".format(counter))


X_train, y_train, X_test, y_test = load_cifar_10(unpack=True)
for seed in range(1000, 1010):
    np.random.seed(seed)
    X_train_, y_train_, mask_train_, X_test_, y_test_, train_file_name_ = prepare_dataset(X_train, y_train, X_test, y_test, 10)
    produce_labeled_filename(X_train_, y_train_, mask_train_, train_file_name_, seed)


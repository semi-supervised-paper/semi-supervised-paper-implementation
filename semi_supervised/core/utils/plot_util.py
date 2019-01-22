import numpy as np
import torch
from ..model.synthetic_net import SyntheticNet

model_location = r'semi_supervised\result'
data_dir = r'semi_supervised\data_local\synthetic'

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

np.random.seed(0)

epochs_to_plot = [5, 10, 20, 30, 40, 50, 100, 200, 300, 400, -1]


def load_two_moons():
    import pickle
    path = os.path.join(data_dir, 'two_moons.save')
    f = open(path, 'rb')
    loaded_objects = []
    for i in range(3):
        loaded_objects.append(pickle.load(f, encoding='latin1'))
    f.close()

    x_train = np.reshape(loaded_objects[0], [-1, 2])
    x_test = np.reshape(loaded_objects[0], [-1, 2])
    y_train = np.int32(loaded_objects[1])
    y_test = np.int32(loaded_objects[1])
    mask = loaded_objects[2]

    return np.float32(x_train), y_train, np.float32(x_test), y_test, mask


def load_four_spins():
    import pickle
    path = os.path.join(data_dir, 'four_spins.save')
    f = open(path, 'rb')
    loaded_objects = []
    for i in range(3):
        loaded_objects.append(pickle.load(f, encoding='latin1'))
    f.close()

    x_train = np.reshape(loaded_objects[0], [-1, 2])
    x_test = np.reshape(loaded_objects[0], [-1, 2])
    y_train = np.int32(loaded_objects[1])
    y_test = np.int32(loaded_objects[1])
    mask = loaded_objects[2]

    return np.float32(x_train), y_train, np.float32(x_test), y_test, mask


def validate(model, X_test, Y_test):
    input_data = []
    output_data = []
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for input, target in zip(X_test, Y_test):
            input_var = torch.autograd.Variable(torch.from_numpy(input))
            input_var = input_var.view(1, input_var.size(0)).float()

            # compute output
            output, _ = model(input_var)

            input_data.append(input_var.data.numpy()[0])
            _, output_label = torch.max(output, 1)
            output_data.append(output_label.item())

    return input_data, output_data


def build_model_baseline(load_network_filename, X_test, Y_test):
    model = SyntheticNet(num_classes=4)

    def load_checkpoint(filepath):
        if os.path.isfile(filepath):
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath)
            start_epoch = checkpoint['epoch']
            best_top1_validate = checkpoint['best_top1_validate']
            best_top5_validate = checkpoint['best_top5_validate']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) "
                  "best_top1_validate = {}, best_top5_validate = {}, "
                  "top1_validate = {}, top5_validate = {}, class_loss_validate = {}, "
                  "class_loss_train = {}, pi_loss_train = {}"
                  .format(filepath, checkpoint['epoch'], best_top1_validate, best_top5_validate,
                          checkpoint['top1_validate'], checkpoint['top5_validate'], checkpoint['class_loss_validate'],
                          checkpoint['class_loss_train'], checkpoint['pi_loss_train']))
        else:
            print("=> no checkpoint found at '{}'".format(filepath))

    load_checkpoint(filepath=load_network_filename)

    return validate(model, X_test, Y_test)


def plot(mesh_X, mesh_Y, X, Y, mask_value, figname):
    import matplotlib.pyplot as plt
    colors = np.array([x for x in 'bcrymykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    plt.scatter(mesh_X[:, 0], mesh_X[:, 1], color=colors[mesh_Y].tolist(), alpha=0.003,
                s=10)
    plt.scatter(X[:, 0], X[:, 1], color=colors[Y].tolist(), alpha=0.05,
                s=10)

    mask_indices = np.where(mask_value == 1)
    input_points_value_mask = X[mask_indices]
    output_points_value_mask = Y[mask_indices]

    plt.scatter(input_points_value_mask[:, 0], input_points_value_mask[:, 1], marker='x',
                color=['k'] * np.shape(output_points_value_mask)[0], alpha=1, s=10)

    plt.savefig(os.path.join(model_location, '{}.png'.format(figname)))
    plt.clf()
    print('save fig {}.png\n'.format(figname))


def plot_points(mesh_X, mesh_Y, X, Y, mask_value, figname):
    import matplotlib.pyplot as plt
    colors = np.array([x for x in 'bcrymykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    plt.scatter(X[:, 0], X[:, 1], color=colors[Y].tolist(), alpha=0.5,
                s=10)

    mask_indices = np.where(mask_value == 1)
    input_points_value_mask = X[mask_indices]
    output_points_value_mask = Y[mask_indices]

    plt.scatter(input_points_value_mask[:, 0], input_points_value_mask[:, 1], marker='x',
                color=['k'] * np.shape(output_points_value_mask)[0], alpha=1, s=10)

    plt.savefig(os.path.join(model_location, '{}_points.png'.format(figname)))
    plt.clf()
    print('save fig {}_points.png\n'.format(figname))


def plot_boundary(mesh_X, mesh_Y, X, Y, mask_value, figname):
    import matplotlib.pyplot as plt
    colors = np.array([x for x in 'bcrymykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    plt.scatter(mesh_X[:, 0], mesh_X[:, 1], color=colors[mesh_Y].tolist(), alpha=0.5,
                s=10)

    plt.savefig(os.path.join(model_location, '{}_boundary.png'.format(figname)))
    plt.clf()
    print('save fig {}_boundary.png\n'.format(figname))


def save_to_plot(model_location, model_name):
    colors = np.array([x for x in 'byrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    X_all, Y_all, _, __, mask = load_four_spins()

    X_all = np.float32(X_all)
    Y_all = np.array(Y_all)

    input_value, output_value = build_model_baseline(os.path.join(model_location, model_name),
                                                     X_all, Y_all)
    input_value = np.reshape(input_value, [-1,2])
    output_value = np.reshape(output_value, [-1])

    X_points = input_value
    Y_points = output_value

    x_min = np.min(X_all[:,0])
    x_max = np.max(X_all[:,0])
    x_mid = (x_max + x_min)/2

    y_min = np.min(X_all[:,1])
    y_max = np.max(X_all[:,1])
    y_mid = (y_max + y_min)/2

    x_min += 0.5 * (x_min - x_mid)
    x_max += 0.5 * (x_max - x_mid)
    y_min += 0.5 * (y_min - y_mid)
    y_max += 0.5 * (y_max - y_mid)

    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_mesh = np.vstack([xx.reshape(-1),yy.reshape(-1)]).T
    X_mesh = np.reshape(X_mesh, [-1, 2, 1, 1])
    Y_mesh = np.zeros_like(X_mesh)
    # this is a hack we should get rid of

    input_value, output_value = build_model_baseline(os.path.join(model_location, model_name),
                                                     X_mesh, Y_mesh)
    input_value = np.reshape(input_value, [-1, 2])
    output_value = np.reshape(output_value, [-1])

    # np.save(os.path.join(model_location, model_name+'_input_mesh'),input_value)
    # np.save(os.path.join(model_location, model_name+'_output_mesh'), output_value)

    return input_value, output_value, X_points, Y_points, mask


def sample_inputs(input, target):
    img_count = 32
    num_img = 10
    label_image = np.zeros((3, 32 * num_img, 32 * img_count))
    count = [0] * 11
    max_count = 100
    for i in range(target.size(0)):
        label = target[i].item() + 1
        if count[label] < max_count:
            if count[label] < img_count and label < num_img:
                label_image[:, label * 32 : (label + 1) * 32, count[label] * 32 : (count[label] + 1) * 32] = input[i].data.numpy()
        count[label] += 1
        
        def save_image(filename, img):
            import scipy
            if len(img.shape) == 3:
                if img.shape[0] == 1:            
                    img = img[0] # CHW -> HW (saves as grayscale)
                else:            
                    img = np.transpose(img, (1, 2, 0)) # CHW -> HWC (as expected by toimage)
            scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(filename)
        save_image('labeled_inputs.png', label_image)
        c = 0
        for i in range(11):
            if count[i] >= max_count:
                c += 1
        if c == 11:
            exit()
        continue


def test():
    model_name = 'checkpoint.ckpt'
    tmp_name = model_name
    mesh_X, mesh_Y, X, Y, mask_value = save_to_plot(model_location, model_name)
    plot(mesh_X, mesh_Y, X, Y, mask_value, tmp_name)
    plot_points(mesh_X, mesh_Y, X, Y, mask_value, tmp_name)
    plot_boundary(mesh_X, mesh_Y, X, Y, mask_value, tmp_name)

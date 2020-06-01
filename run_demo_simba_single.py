from keras import utils, Model
from keras.datasets import cifar10
from keras.layers import Input
import numpy as np
import logging
import argparse
from art.classifiers import KerasClassifier
from art.attacks import SimBA
import vgg_model
import matplotlib.pyplot as plt

# SimBA parameters
parser = argparse.ArgumentParser(description='run SimBA demo')
parser.add_argument('--attack', type=str, default='dct', help='attack type: pixel (px) or DCT (dct) attacks')
parser.add_argument('--epsilon', type=float, default=0.1, help='overshoot parameter')
parser.add_argument('--order', type=str, default='random', help='order of pixel attacks: random or diag (diagonal)')
parser.add_argument('--freq_dim', type=int, default=4, help='dimensionality of 2D frequency space')
parser.add_argument('--max_iter', type=int, default=1000, help='maximum number of iterations')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
args = parser.parse_args()

# configure a logger to capture outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# load CIFAR-10 image data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# select test images
idx_sample_imgs = np.array([8,12])
x_test = x_test[idx_sample_imgs]
y_test = y_test[idx_sample_imgs]

# normalize data
channel_mean = np.mean(x_train, axis=(0, 1, 2))
channel_std = np.std(x_train, axis=(0, 1, 2))
for i in range(3):
    x_test[:, :, :, i] = (x_test[:, :, :, i] - channel_mean[i]) / channel_std[i]

# labels to categorical
num_classes = 10
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# build vgg model
num_blocks = 3
img_input = Input(shape=(32, 32, 3), name='input')
img_prediction = vgg_model.vgg_model(img_input, 10, num_blocks)
model = Model(img_input, img_prediction)
model.load_weights("model/vgg_20_1553198546.h5")
classifier = KerasClassifier(model=model)

# set CIFAR-10 labels
label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# set the targeted label (ship, in this case)
if args.targeted:
    targeted_y = np.zeros(10)
    targeted_y[8] = 1.0

# normalization for image plot
def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

### SimBA ###########
fig, ax = plt.subplots(len(idx_sample_imgs), 3)
for i in range(len(idx_sample_imgs)):
    logger.info('Craft attack on an image with SimBA')
    adv_crafter = SimBA(classifier, attack=args.attack, order=args.order, epsilon=args.epsilon, max_iter=args.max_iter, freq_dim=args.freq_dim, targeted=args.targeted)
    if args.targeted:
        x_test_adv = adv_crafter.generate(x_test[i].reshape(1,32,32,3), y=targeted_y.reshape(1,10))
    else:
        x_test_adv = adv_crafter.generate(x_test[i].reshape(1,32,32,3))
        #x_test_adv = adv_crafter.generate(x_test[i].reshape(1,32,32,3), y=y_test[i].reshape(1,10))
    
    preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)

    logger.info('Plot the orignail image, perturbation, and adversarial image')
    ax[i][0].imshow(norm(x_test[i]))
    plt.axis('off')
    ax[i][0].set_title(label[np.argmax(y_test[i])])
    ax[i][1].imshow(norm(x_test_adv[0] - x_test[i]))
    plt.axis('off')
    ax[i][1].set_title("perturbation")
    ax[i][2].imshow(norm(x_test_adv[0]))
    plt.axis('off')
    ax[i][2].set_title(label[preds_adv[0]])

plt.savefig('plot_simba_demo.png')

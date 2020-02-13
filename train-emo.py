import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from utils.dataprocess import shorten
from utils.helpers import save_hist, one_hot
from net.datasets import DynamicPaddingEmoSet
from net.models import AudioEmotion
from net.callbacks import EarlyStopping

DATA_PATH = r'/home/wu/mounts/Emo-gesture/train_set.pkl'

# settings
RATIO = 1
BATCH_SIZE = 32
SEED = 1234567890
VALID_SIZE = 0.2

EPOCHS = 200
LR = 1e-4
PATIENCE = 200

MODEL_NAME = 'chkpt-emo'


def load_data():
    audio_data, motion_data, emo_data = pickle.load(open(DATA_PATH, 'rb'))
    emo_data = list(map(one_hot, emo_data))

    X_train, X_valid, e_train, e_valid, Y_train, Y_valid = train_test_split(
        audio_data, emo_data, motion_data, test_size=VALID_SIZE, random_state=SEED)

    X_train, Y_train = shorten(X_train, Y_train)
    Y_train = list(map(lambda x: x.reshape(x.shape[0], -1), Y_train))
    X_valid, Y_valid = shorten(X_valid, Y_valid)
    Y_valid = list(map(lambda x: x.reshape(x.shape[0], -1), Y_valid))
    return X_train, e_train, Y_train, X_valid, e_valid, Y_valid


if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("Device: ", device)

    X_train, e_train, Y_train, X_valid, e_valid, Y_valid = load_data()
    train_set = DynamicPaddingEmoSet(
        X_train, e_train, Y_train, batch_size=BATCH_SIZE)
    valid_set = DynamicPaddingEmoSet(
        X_valid, e_valid, Y_valid, batch_size=BATCH_SIZE)

    net = AudioEmotion(hidden_dim=128, x_input_dim=20, e_input_dim=11, output_dim=30,
                       num_stack=2, return_sequences=True, return_states=False)
    print(net)
    print("Num of parameters: ", net.count_parameters())
    net.compile(lr=LR, optimizer='adam', criterion='mse', device=device)
    stopping = EarlyStopping(patience=PATIENCE, verbose=True, name=MODEL_NAME)
    hist = net.fit_generator(train_set, valid_set,
                             epochs=EPOCHS, stopping=stopping)
    save_hist(hist, ['train_loss', 'valid_loss'],
              MODEL_NAME, plot=True, csv=True)

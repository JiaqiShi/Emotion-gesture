
import pickle
import torch
from sklearn.model_selection import train_test_split
from utils.helpers import shorten
from net.datasets import DynamicPaddingSet
from net.models import AudioOnly
from net.callbacks import EarlyStopping

DATA_PATH = r'/home/wu/mounts/Emo-gesture/train_set.pkl'

# settings
RATIO = 1
BATCH_SIZE = 64
SEED = 1234567890
VALID_SIZE = 0.2

EPOCHS = 100
PATIENCE = 15


def load_data():
    audio_data, motion_data, _ = pickle.load(open(DATA_PATH, 'rb'))
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        audio_data, motion_data, test_size=VALID_SIZE, random_state=SEED)
    
    X_train, Y_train = shorten(X_train, Y_train)
    Y_train = list(map(lambda x: x.reshape(x.shape[0], -1), Y_train))
    X_valid, Y_valid = shorten(X_valid, Y_valid)
    Y_valid = list(map(lambda x: x.reshape(x.shape[0], -1), Y_valid))
    return X_train, Y_train, X_valid, Y_valid

if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("Device: ", device)
    
    X_train, Y_train, X_valid, Y_valid = load_data()
    train_set = DynamicPaddingSet(X_train, Y_train, batch_size=BATCH_SIZE)
    valid_set = DynamicPaddingSet(X_valid, Y_valid, batch_size=BATCH_SIZE)

    net = AudioOnly(hidden_dim=128, input_dim=20, output_dim=30,
                    num_stack=2, return_sequences=True, return_states=False).to(device)
    print(net)
    print("Num of parameters: ", net.count_parameters())
    net.compile(lr=1e-4, optimizer='adam', criterion='mse')
    stopping = EarlyStopping(patience=PATIENCE, verbose=True, name='chkpt.pt')
    net.fit_generator(train_set, valid_set, epochs=EPOCHS, device=device, stopping=stopping)


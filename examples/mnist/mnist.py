import click
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
import uuid

import sys

from ptdec.dec import DEC
from ptdec.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy


class CachedMNIST(Dataset):
    def __init__(self, data_dir, is_train, device, testing_mode=False):
        img_transform = transforms.Compose([
            transforms.Lambda(self._transformation)
        ])

        self.ds = MNIST(
            data_dir,
            download=True,
            train=is_train,
            transform=img_transform,
        )

        self.device = device
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float() * 0.02

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)

def main(
    data_dir,
    cuda,
    batch_size,
    pretrain_epochs,
    finetune_epochs,
    testing_mode
):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars('data/autoencoder', {
            'lr': lr,
            'loss': loss,
            'validation_loss': validation_loss,
        }, epoch)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_train = CachedMNIST(data_dir, is_train=True, device=device, testing_mode=testing_mode)  # training dataset
    ds_val = CachedMNIST(data_dir, is_train=False, device=device, testing_mode=testing_mode)  # evaluation dataset
    autoencoder = StackedDenoisingAutoEncoder(
        [28 * 28, 500, 500, 2000, 10],
        final_activation=None
    )

    autoencoder = autoencoder.to(device)
    print('Pretraining stage.')
    ae.pretrain(
        ds_train,
        autoencoder,
        device=device,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 20000, gamma=0.1),
        corruption=0.2
    )
    print('Training stage.')
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        device=device,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 20000, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback
    )
    print('DEC stage.')
    model = DEC(
        cluster_number=10,
        embedding_dimension=28 * 28,
        hidden_dimension=10,
        encoder=autoencoder.encoder
    )

    model = model.to(device)
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=1000,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda
    )
    predicted, actual = predict(ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda)
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print('Final DEC accuracy: %s' % accuracy)
    if not testing_mode:
        predicted_reassigned = [reassignment[item] for item in predicted]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion).get_figure().savefig('confusion_%s.png' % confusion_id)
        print('Writing out confusion diagram with UUID: %s' % confusion_id)
        writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DEC.')
    parser.add_argument('--data_dir', required=True, type=str, help='Root directory contains training/testing data')
    parser.add_argument('--cuda', type=bool, default=False, help='Whether to use cuda')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--pretrain', type=int, default=50000, help='# of pretrain epoches')
    parser.add_argument('--finetune', type=int, default=100000, help='# of finetune epoches')
    parser.add_argument('--testing_mode', type=bool, default=True, help='Testing mode')

    args = parser.parse_args()
    main(args.data_dir,
        args.cuda,
        args.batch_size,
        args.pretrain,
        args.finetune,
        args.testing_mode)

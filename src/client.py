import gc
import pickle
import logging
import sys

import wandb

import torch
import torch.nn as nn

# from torch.utils.data import DataLoader
from .DatasetLoader import get_data_loader, get_data_loader_speaker

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        # return len(self.data)
        return len(iter(self.dataloader))

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        # self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.dataloader = get_data_loader(dataset_file_name='/nvme/zhiyong/sdsv21/vox2_trainlist.txt', batch_size=128, augment=False, musan_path='/nvme/zhiyong/musan_split', rir_path='/nvme/zhiyong/RIRS_NOISES/simulated_rirs', max_frames=300, max_seg_per_spk=100, nDataLoaderThread=8, nPerSpeaker=1, train_path='/nvme/zhiyong/sdsv21', sox_aug=False)
        self.local_epoch = client_config["num_local_epochs"]
        # self.criterion = client_config["criterion"]
        # self.optimizer = client_config["optimizer"]
        # self.optim_config = client_config["optim_config"]

    def setup_speaker_client(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        # self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        print("ID:", self.id)
        self.local_epoch = client_config["num_local_epochs"]
        # self.criterion = client_config["criterion"]
        # self.optimizer = client_config["optimizer"]
        # self.optim_config = client_config["optim_config"]

        self.dataset_file = client_config["multi_dataset_files"][self.id]
        self.train_path = client_config["multi_train_paths"][self.id]

        self.dataloader = get_data_loader_speaker(self.id*100, dataset_file_name=self.dataset_file, 
        batch_size=128, augment=False, musan_path='/nvme/zhiyong/musan_split', 
        rir_path='/nvme/zhiyong/RIRS_NOISES/simulated_rirs', max_frames=300, 
        max_seg_per_spk=100, nDataLoaderThread=8, nPerSpeaker=1, 
        train_path=self.train_path, sox_aug=False)

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)

        # optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        optimizer = torch.optim.SGD(self.model.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=5e-4)
        for e in range(self.local_epoch):
            loss_total = 0
            counter = 0
            for data, data_label in self.dataloader:
                # data, labels = data.float().to(self.device), labels.long().to(self.device)
                data    = data.transpose(1,0)
                label   = torch.LongTensor(data_label).cuda()

                optimizer.zero_grad()
                # outputs = self.model(data)
                # loss = eval(self.criterion)()(outputs, labels)
                loss, prec = self.model(data, label)

                loss.backward()
                optimizer.step()

                loss_total += loss.detach().cpu().numpy()
                counter += 1
                sys.stdout.write("Loss %f \r"%(loss_total/counter))
                sys.stdout.flush()

                wandb.log({"C%dloss"%self.id: loss})

                if self.device == "cuda": torch.cuda.empty_cache()

        self.model.to("cpu")

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy

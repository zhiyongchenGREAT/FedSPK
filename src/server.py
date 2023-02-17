import copy
import gc
import logging

import numpy as np
import torch
import torch.nn as nn

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict
import time, random, sys, numpy

from .models import *
from .utils import *
from .client import Client

from .SpeakerNet import SpeakerNet
from .tuneThreshold import tuneThresholdfromScore_std
from .DatasetLoader import loadWAV

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        # self.model = eval(model_config["name"])(**model_config)
        if model_config["use_additional_model"]:
            self.model = SpeakerNet(model='X_vector', trainfunc='softmax', nPerSpeaker=1, Syncbatch=False, n_mels=40, nOut=192, spec_aug=False, nClasses=5994, additional_model=True)
        else:
            self.model = SpeakerNet(model='X_vector', trainfunc='softmax', nPerSpeaker=1, Syncbatch=False, n_mels=40, nOut=192, spec_aug=False, nClasses=5994)
        
        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

        # self.data_path = data_config["data_path"]
        # self.dataset_name = data_config["dataset_name"]
        # self.num_shards = data_config["num_shards"]
        # self.iid = data_config["iid"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]

        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config
        
        self.multi_dataset_files = data_config["multi_dataset_files"]
        self.multi_train_paths = data_config["multi_train_paths"]

        self.speaker_init_model = init_config["speaker_init_model"]

    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)
        # init_net(self.model, **self.init_config)
        self.model.to(self.init_config['gpu_ids'][0])

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        # local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid)
        

        # assign dataset to each client
        # self.clients = self.create_clients(local_datasets)
        self.clients = self.create_clients(None)

        # prepare hold-out dataset for evaluation
        # self.data = test_dataset
        # self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config
            )
        
        # send the model skeleton to all clients
        self.transmit_model()
    
    def setup_speaker_server(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)
        # init_net(self.model, **self.init_config)
        self.model.to(self.init_config['gpu_ids'][0])

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        # local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid)
        

        # assign dataset to each client
        # self.clients = self.create_clients(local_datasets)
        self.clients = self.create_clients(None)

        # prepare hold-out dataset for evaluation
        # self.data = test_dataset
        # self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # configure detailed settings for client upate and 
        self.setup_speaker_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config,
            multi_dataset_files=self.multi_dataset_files,
            multi_train_paths = self.multi_train_paths
            )
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        # for k, dataset in tqdm(enumerate(local_datasets), leave=False):
        for k in tqdm(range(self.num_clients), leave=False):
            # client = Client(client_id=k, local_data=dataset, device=self.device)
            client = Client(client_id=k, local_data=None, device=self.device)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def setup_speaker_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup_speaker_client(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()
    
    def transmit_model_onlyparam(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                self.load_paras_to_clients(self.model, client.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.load_paras_to_clients(self.model, self.clients[idx].model)

            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size

    def update_selected_clients_speaker(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update()
        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()

    def average_model_speaker(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        averaged_weights = OrderedDict()
        
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if '__L__' in key:
                    continue
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]

        self.model.load_state_dict(averaged_weights, strict=False)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    # def evaluate_selected_models(self, sampled_client_indices):
    #     """Call "client_evaluate" function of each selected client."""
    #     message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
    #     print(message); logging.info(message)
    #     del message; gc.collect()

    #     for idx in sampled_client_indices:
    #         self.clients[idx].client_evaluate()

    #     message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
    #     print(message); logging.info(message)
    #     del message; gc.collect()

    # def evaluate_selected_models(self, indice):
    #     # evaluate the selected client


    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # # evaluate selected clients with local dataset (same as the one used for local update)
        # if self.mp_flag:
        #     message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        #     print(message); logging.info(message)
        #     del message; gc.collect()

        #     with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
        #         workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        # else:
        #     self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)

    def train_federated_model_speaker(self):
        """Do federated training."""

        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model_onlyparam(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_selected_clients_speaker(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        # self.average_model(sampled_client_indices, mixing_coefficients)
        self.average_model_speaker(sampled_client_indices, mixing_coefficients)
        
    # def evaluate_global_model(self):
    #     """Evaluate the global model using the global holdout dataset (self.data)."""
    #     self.model.eval()
    #     self.model.to(self.device)

    #     test_loss, correct = 0, 0
    #     with torch.no_grad():
    #         for data, labels in self.dataloader:
    #             data, labels = data.float().to(self.device), labels.long().to(self.device)
    #             outputs = self.model(data)
    #             test_loss += eval(self.criterion)()(outputs, labels).item()
                
    #             predicted = outputs.argmax(dim=1, keepdim=True)
    #             correct += predicted.eq(labels.view_as(predicted)).sum().item()
                
    #             if self.device == "cuda": torch.cuda.empty_cache()
    #     self.model.to("cpu")

    #     test_loss = test_loss / len(self.dataloader)
    #     test_accuracy = correct / len(self.data)
    #     return test_loss, test_accuracy

    def load_paras_to_clients(self, model, cli_model):
        self_state = cli_model.state_dict()
        loaded_state = model.state_dict()

        for name, param in loaded_state.items():
            if '__L__' in name:
                continue

            if name not in self_state:
                print("#%s is not in the model."%name)
                continue

            if self_state[name].size() != loaded_state[name].size():
                print("#Wrong parameter length: %s, model: %s, loaded: %s"%(name, self_state[name].size(), loaded_state[name].size()))
                continue

            self_state[name].copy_(param)       



    def loadParameters(self, model, path, map_location="cuda:0"):

        self_state = model.state_dict()
        loaded_state = torch.load(path, map_location=map_location)

        for name, param in loaded_state['model'].items():
            if '__L__' in name:
                continue
            
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    name = "__S__."+name
                    if name not in self_state:
                        print("#%s is not in the model."%origname)
                        continue

            if self_state[name].size() != loaded_state['model'][origname].size():
                print("#Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state['model'][origname].size()))
                continue

            self_state[name].copy_(param)

    def evaluateFromList(self, model, listfilename, distance_m='cosine', print_interval=100, test_path='', num_eval=10, eval_frames=0, verbose=True):
        assert distance_m in ['L2', 'cosine']
        if verbose:
            print('Distance metric: %s'%(distance_m))
            print('Evaluating from trial file: %s'%(listfilename))
        
        model = model.cuda()

        model.eval()
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline()
                if (not line):
                    break

                data = line.split()

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()

            ref_feat = model.forward(inp1).detach().cpu()

            filename = '%06d.wav'%idx

            feats[file] = ref_feat

            telapsed = time.time() - tstart

            if (idx % print_interval == 0) and verbose:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx, len(setfiles), idx/telapsed, ref_feat.size()[1]))

        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data


            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]
            # ref_feat = (feats[data[1]] - mean_vector).cuda() 
            # com_feat = (feats[data[2]] - mean_vector).cuda()

            # if self.__model__.module.__L__.test_normalize:
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)

            if distance_m == 'L2':
                dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy()
                score = -1 * numpy.mean(dist)
            elif distance_m == 'cosine':
                ## [1, emb_size]
                dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy()
                score = numpy.mean(dist)
            else:
                raise ValueError('Unknown distance metric: %s'%(distance_m))

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1]+" "+data[2])

            if (idx % (print_interval*100) == 0) and verbose:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed))
                sys.stdout.flush()

        result = tuneThresholdfromScore_std(all_scores, all_labels)
        print('')
        print('EER %2.4f MINC@0.01 %.5f MINC@0.001 %.5f'%(result[1], result[-2], result[-1]))

    def evaluate_global_model(self, task, model_path, listfilename, testfile_path):
        if task == 'ver':
            self.loadParameters(self.model, model_path)
            self.evaluateFromList(self.model, listfilename, test_path=testfile_path)
        else:
            raise ValueError('Unknown task: %s'%(task))

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}

        for r in range(self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()
            # test_loss, test_accuracy = self.evaluate_global_model()
            
            # self.results['loss'].append(test_loss)
            # self.results['accuracy'].append(test_accuracy)

            # self.writer.add_scalars(
            #     'Loss',
            #     {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
            #     self._round
            #     )
            # self.writer.add_scalars(
            #     'Accuracy', 
            #     {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy},
            #     self._round
            #     )

            # message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
            #     \n\t[Server] ...finished evaluation!\
            #     \n\t=> Loss: {test_loss:.4f}\
            #     \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            # print(message); logging.info(message)
            # del message; gc.collect()
        self.transmit_model()

    def fit_speaker(self):
        """Execute the whole process of the federated learning."""
        # self.results = {"loss": [], "accuracy": []}
        # init speaker model with existing parameters
        self.loadParameters(self.model, self.speaker_init_model)
        
        for r in range(self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model_speaker()
            # test_loss, test_accuracy = self.evaluate_global_model()
            
            # self.results['loss'].append(test_loss)
            # self.results['accuracy'].append(test_accuracy)

            # self.writer.add_scalars(
            #     'Loss',
            #     {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
            #     self._round
            #     )
            # self.writer.add_scalars(
            #     'Accuracy', 
            #     {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy},
            #     self._round
            #     )

            # message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
            #     \n\t[Server] ...finished evaluation!\
            #     \n\t=> Loss: {test_loss:.4f}\
            #     \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            # print(message); logging.info(message)
            # del message; gc.collect()
        self.transmit_model_onlyparam()

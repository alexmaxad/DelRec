import torch 
import numpy as np
import h5py
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.distributions.binomial import Binomial
from typing import Callable, Optional

from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets import pad_sequence_collate

from src.utils import seed_everything

def load_dataset(config):
    if config.dataset == 'SHD':
        return SHD_dataloaders(config)
    elif config.dataset == 'SSC':
       return SSC_dataloaders(config)
    elif config.dataset == 'PSMNIST':
        return PS_MNIST_dataloaders(config)
    else:
        raise ValueError(f"Dataset {config.dataset} is not supported.")

def SHD_dataloaders(config):
  seed_everything(config.seed,is_cuda=True)
  
  train_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True, data_type='frame', duration=config.time_step)
  test_dataset= BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=False, data_type='frame', duration=config.time_step)

  train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])
  
  if config.use_augmentations:
        train_dataset = SHDTripleAugDataset(train_dataset, shift_max=config.shift_max, thin_p=config.thin_p, jitter_in_blend=config.jitter_in_blend)

  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  return train_loader, valid_loader, test_loader

class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
            self,
            root: str,
            n_bins: int,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            if self.transform is not None:
                binned_frames = self.transform(binned_frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return binned_frames, label
        
class SHDTripleAugDataset(Dataset):
    def __init__(self,
                 base_ds_or_subset,
                 shift_max: int = 40,
                 thin_p: float = 0.5,
                 jitter_in_blend: bool = False):
        super().__init__()
        self.base = base_ds_or_subset
        self.shift_max = int(shift_max)
        self.thin_p = float(thin_p)
        self.jitter_in_blend = bool(jitter_in_blend)
        
        self.is_subset = hasattr(self.base, "indices") and hasattr(self.base, "dataset")
        if self.is_subset:
            self.indices = list(self.base.indices)
        else:
            self.indices = list(range(len(self.base)))
        self.n = len(self.indices)

        self.labels = np.empty(self.n, dtype=np.int64)
        for pos in range(self.n):
            _, y = self._get_item_pos(pos)
            self.labels[pos] = int(y)

        self.class_pos = {}
        for pos, y in enumerate(self.labels):
            self.class_pos.setdefault(int(y), []).append(pos)
            
    def _get_item_pos(self, pos: int):
        if self.is_subset:
            return self.base[pos]                   
        else:
            return self.base[self.indices[pos]] 

    @staticmethod
    def _time_shift_per_neuron(x: torch.Tensor, shift_max: int) -> torch.Tensor:
        if shift_max <= 0:
            return x
        T, N = x.shape
        out = x.new_zeros(T, N)
        shifts = torch.randint(-shift_max, shift_max + 1, (N,), device=x.device)
        for n in range(N):
            s = int(shifts[n])
            if s > 0:
                # delay by s
                L = T - s
                if L > 0:
                    out[s:s+L, n] = x[:L, n]
            elif s < 0:
                # advance by |s|
                s2 = -s
                L = T - s2
                if L > 0:
                    out[:L, n] = x[s2:s2+L, n]
            else:
                out[:, n] = x[:, n]
        return out

    @staticmethod
    def _com_time(x: torch.Tensor) -> float:
        T = x.shape[0]
        mass_t = x.sum(dim=1)
        denom = mass_t.sum()
        if denom <= 0:
            return 0.5 * (T - 1)
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        return float((t * mass_t).sum() / denom)

    @staticmethod
    def _thin_binomial(x: torch.Tensor, p: float) -> torch.Tensor:
        xi = x.clamp_min(0).round()
        if xi.numel() == 0:
            return xi
        dist = Binomial(total_count=xi, probs=torch.tensor(p, device=xi.device))
        return dist.sample()
    
    @staticmethod
    def _align_and_pad_pair(x: torch.Tensor, xb: torch.Tensor, align: int) -> tuple[torch.Tensor, torch.Tensor]:
        T1, N = x.shape
        T2, N2 = xb.shape
        assert N == N2, "Neuron/channel dimension mismatch."

        pad_x_left  = max(0, -align)   # if align<0, push x to the right
        pad_xb_left = max(0,  align)   # if align>0, push xb to the right

        T_out = max(T1 + pad_x_left, T2 + pad_xb_left)

        x_pad  = x.new_zeros(T_out, N)
        xb_pad = xb.new_zeros(T_out, N)

        x_pad[ pad_x_left : pad_x_left + T1, : ] = x
        xb_pad[pad_xb_left: pad_xb_left + T2, : ] = xb

        return x_pad, xb_pad

    def _sample_same_class_partner(self, y: int, avoid_pos: int) -> Optional[int]:
        pool = self.class_pos.get(int(y), [])
        if len(pool) <= 1:
            return None
        j = avoid_pos
        while j == avoid_pos:
            j = pool[np.random.randint(0, len(pool))]
        if int(self.labels[j]) != int(y):
            return None
        return j

    def __len__(self) -> int:
        return 3 * self.n

    def __getitem__(self, idx: int):
        
        if idx < self.n: 
            x, y = self._get_item_pos(idx)            # (T, N), y
            return torch.as_tensor(x, dtype=torch.float32), int(y)

        if idx < 2 * self.n:
            pos = idx - self.n
            x, y = self._get_item_pos(pos)
            x = torch.as_tensor(x, dtype=torch.float32)
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        pos = idx - 2 * self.n
        x, y = self._get_item_pos(pos)
        x = torch.as_tensor(x, dtype=torch.float32)

        partner_pos = self._sample_same_class_partner(int(y), pos)
        if partner_pos is None:
            # no partner ->  shift-only augmentation
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        xb, yb = self._get_item_pos(partner_pos)
        xb = torch.as_tensor(xb, dtype=torch.float32)
        if int(yb) != int(y):
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        align = int(round(self._com_time(x) - self._com_time(xb)))
        x, xb = self._align_and_pad_pair(x, xb, align)

        if self.jitter_in_blend and self.shift_max > 0:
            x  = self._time_shift_per_neuron(x,  self.shift_max)
            xb = self._time_shift_per_neuron(xb, self.shift_max)

        x = self._thin_binomial(x,  self.thin_p) + self._thin_binomial(xb, self.thin_p)
        return x, int(y)
        
def SSC_dataloaders(config):
    seed_everything(config.seed,is_cuda=True)
    T = config.time_window
    max_time = 1.4
    
    root_path = config.datasets_path
    dataset = config.dataset
    
    train_file = h5py.File(os.path.join(root_path, dataset.lower()+'_train.h5'), 'r')
    valid_file = h5py.File(os.path.join(root_path, dataset.lower()+'_valid.h5'), 'r')
    test_file = h5py.File(os.path.join(root_path, dataset.lower()+'_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_valid = valid_file['spikes']
    y_valid = valid_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    
    train_loader = SSC_SpikeIterator(x_train, y_train, config.batch_size, T, 700, max_time, n_bins = config.n_bins, shuffle=True)
    valid_loader = SSC_SpikeIterator(x_valid, y_valid, config.batch_size, T, 700, max_time, n_bins = config.n_bins, shuffle=True)
    test_loader = SSC_SpikeIterator(x_test, y_test, config.batch_size, T, 700, max_time, n_bins = config.n_bins, shuffle=False)
    
    return train_loader, valid_loader, test_loader

    
class SSC_SpikeIterator:
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, n_bins = 1,shuffle=True, device='cuda:0', indices=None, label_map=None):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.float32)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))
        
        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)

        self.n_bins = n_bins

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.number_of_batches)

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[
                          self.batch_size * self.counter:min(self.batch_size * (self.counter + 1), self.num_samples)]
            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins)
                units = self.units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(self.device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)

            X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
                [len(batch_index), self.nb_steps, self.nb_units])).to_dense().to(
                self.device)
            
            ###############################################################
            binned_len = X_batch.shape[-1]//self.n_bins
            binned_frames = torch.zeros((len(batch_index), self.nb_steps, binned_len)).to(
                self.device)
            for i in range(binned_len):
                binned_frames[:,:,i] = X_batch[:, :,self.n_bins*i : self.n_bins*(i+1)].sum(axis=-1)
            ###############################################################
            y_batch = torch.tensor(self.labels_[batch_index], device=self.device).long()

            X_batch = binned_frames

            self.counter += 1
            return X_batch.to(device=self.device), y_batch.to(device=self.device)

        else:
            raise StopIteration

def PS_MNIST_dataloaders(config):
    seed_everything(config.seed,is_cuda=True)
    config.time_window = 784
    config.input_dim = 1
    config.output_dim = 10
    is_cuda = True
    kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if is_cuda else {}
    dataset_train = datasets.MNIST(config.datasets_path, train=True, download=True,
                                    transform=transforms.ToTensor())
    
    train_size = int(0.83 * len(dataset_train))  # 50,000
    val_size = len(dataset_train) - train_size   # 10,000
    dataset_train, dataset_val = random_split(
        dataset_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader = DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        dataset_val, batch_size=config.batch_size, shuffle=False, **kwargs
    )
    test_loader = DataLoader(
        datasets.MNIST(config.datasets_path, train=False, transform=transforms.ToTensor()),
        batch_size=config.batch_size, shuffle=False, **kwargs
    )
    
    return train_loader, val_loader, test_loader
import torch

from spikingjelly.activation_based import neuron, base,  layer

class recurrent_neuron(base.MemoryModule):
    
    def __init__(
        self,
        config, 
        neurons: int,
        neuron_module: torch.nn.Module = neuron.LIFNode, 
        ):

        super().__init__()
        
        self.config = config
        self.step_mode = config.step_mode
        self.store_v_seq = config.store_v_seq
        
        self.neuron_module = neuron_module(
            tau = config.tau,
            decay_input = config.decay_input,
            v_reset = config.v_reset,
            v_threshold = config.v_threshold,
            surrogate_function = config.surrogate_function,
            detach_reset = config.detach_reset,
            step_mode = 's',
            backend = config.backend,
        )
        
        self.neurons = neurons
        
        self.recurrent_weights = torch.nn.Parameter(torch.zeros(neurons, neurons), requires_grad=True)
        
        self.dropout = layer.Dropout(config.recurrent_dropout_rate, step_mode='s')
        
        self.init_recurrent_weights()
        
    def init_recurrent_weights(self):
        torch.nn.init.orthogonal_(self.recurrent_weights, gain=self.config.rec_delay_init_gain)
        
    def multi_step_forward(self, x_seq: torch.Tensor):
        # x_seq: (T, B, N)
        device = x_seq.device
        dtype  = x_seq.dtype
        T, B, N = x_seq.shape
        y_seq = []

        W = self.recurrent_weights.to(device=device, dtype=dtype).unsqueeze(0) # (1, N_in, N_out)

        if self.store_v_seq:
            v_seq = []

        rec_input = torch.zeros(B, N, device=device, dtype=dtype)
        
        for t in range(T):
            
            # print("rec_now:", rec_input)

            y = self.neuron_module.single_step_forward(x_seq[t] + self.dropout(rec_input))  # (B, N_out)
            
            rec_input = torch.matmul(W, y.unsqueeze(-1)).squeeze(-1) # (B, N_in)

            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.neuron_module.v)

        self.x_seq = x_seq
        
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

class axonal_recdel(base.MemoryModule):
    def __init__(
        self,
        config, 
        neurons: int,
        neuron_module: torch.nn.Module = neuron.LIFNode, 
        ):

        super().__init__()
        
        self.config = config
        self.step_mode = config.step_mode
        self.store_v_seq = config.store_v_seq
        
        self.neuron_module = neuron_module(
            tau = config.tau,
            decay_input = config.decay_input,
            v_reset = config.v_reset,
            v_threshold = config.v_threshold,
            surrogate_function = config.surrogate_function,
            detach_reset = config.detach_reset,
            step_mode = 's',
            backend = config.backend,
        )
        
        self.neurons = neurons
        
        self.sigma = float(self.config.sigma_init)
        
        # For sanity check :
        # self.sigma = 0
        
        self.recurrent_weights = torch.nn.Parameter(torch.zeros(neurons, neurons), requires_grad=True)
        self.recurrent_delays = torch.nn.Parameter(torch.zeros(neurons), requires_grad=True)
        
        self.dropout = layer.Dropout(config.recurrent_dropout_rate, step_mode='s')
        
        self.use_sig_p = config.use_sig_p
        if self.use_sig_p:
            self.p_spread = torch.nn.Parameter(torch.zeros(neurons), requires_grad=True) 
        
        self.init_recurrent_weights()
        self.init_recurrent_delays()
        
    def init_recurrent_weights(self):
        torch.nn.init.orthogonal_(self.recurrent_weights, gain=self.config.rec_delay_init_gain)

    def init_recurrent_delays(self):
        with torch.no_grad():
            if self.config.init_rec_delay == 'half_normal':
                half_normal = torch.abs(torch.randn_like(self.recurrent_delays) * self.config.delay_std_init)
                self.recurrent_delays.copy_(half_normal)
                self.recurrent_delays.clamp_(min=0.0)
            elif self.config.init_rec_delay == 'uniform':
                torch.nn.init.uniform_(self.recurrent_delays, a=self.config.init_recdel_offset, b=self.config.max_rec_delay)
                self.recurrent_delays.clamp_(min=0.0)
            
    def update_sigma(self, current_epoch):
        decay_per_epoch = self.config.sigma_decay ** (100 / self.config.epochs)
        self.sigma = self.config.sigma_init * (decay_per_epoch ** current_epoch)   
            
    def clamp_recurrent_delays(self):
        with torch.no_grad():
            self.recurrent_delays.clamp_(min=0)
    
    def multi_step_forward(self, x_seq: torch.Tensor):
        # x_seq: (T, B, N)
        device = x_seq.device
        dtype  = x_seq.dtype
        T, B, N = x_seq.shape
        y_seq = []

        W = self.recurrent_weights.to(device=device, dtype=dtype).unsqueeze(0)   # (1, N_in, N_out)
        d = self.recurrent_delays.to(device=device, dtype=dtype)    # (N_out,)

        if self.use_sig_p:
            s = 1.0 + 2.0 * self.sigma * torch.sigmoid(self.p_spread).to(device=device, dtype=dtype) # (N_out,)
            s_max = s.max()
            s = s.unsqueeze(1) # (N_out, 1)
        else:
            s = torch.tensor(1.0 + float(self.sigma), device=device, dtype=dtype)
            s_max = s
            
        L = int(torch.ceil(1.0 + torch.max(d) + s_max).item()) + 1
        support = torch.arange(L, device=device, dtype=dtype)              # (L,)

        mask = (torch.clamp(s - torch.abs(support[None, :] - (1.0 + d)[:, None]), min=0.0) / (s * s)).unsqueeze(0) # (1, N_out, L)

        buffer = torch.zeros(B, N, L, device=device, dtype=dtype)
        pointer = 0

        if self.store_v_seq:
            v_seq = []

        for t in range(T):
            rec_now = buffer[:, :, pointer]                       # (B, N_in)
    
            # print("rec_now:", rec_now)

            y = self.neuron_module.single_step_forward(x_seq[t] + self.dropout(rec_now))  # (B, N_out)
            
            y_masked = y.unsqueeze(2) * mask # (B, N_out, L)
            X_rec = torch.matmul(W, y_masked) # (B, N_in, L)
            
            buffer[:, :, pointer] = 0.0
            pointer = (pointer + 1) % L

            first_chunk = min(L - 1, L - pointer)
            if first_chunk > 0:
                buffer[:, :, pointer:pointer + first_chunk].add_(X_rec[:, :, 1:1 + first_chunk])
            remaining = L - 1 - first_chunk
            if remaining > 0:
                buffer[:, :, 0:remaining].add_(X_rec[:, :, 1 + first_chunk:1 + first_chunk + remaining])

            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.neuron_module.v)

        self.x_seq = x_seq
        
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def forward(self, x_seq: torch.Tensor):
        if self.step_mode == 's':
            return super().single_step_forward(x_seq)
        else:
            return self.multi_step_forward(x_seq)
        
class synaptic_recdel(axonal_recdel):
    def __init__(
        self,
        config, 
        neurons: int,
        neuron_module: torch.nn.Module = neuron.LIFNode, 
        ):

        super().__init__(
            config=config,
            neurons=neurons,
            neuron_module=neuron_module,
        )
        
        # For sanity check :
        # self.sigma = 0
        
        self.recurrent_delays = torch.nn.Parameter(torch.zeros(neurons, neurons), requires_grad=True) # (N, N)
        self.init_recurrent_delays()
        
    def multi_step_forward(self, x_seq: torch.Tensor):
        # x_seq: (T, B, N)
        device = x_seq.device
        dtype  = x_seq.dtype
        T, B, N = x_seq.shape
        y_seq = []

        W = self.recurrent_weights.to(device=device, dtype=dtype)   # (N_in, N_out)
        d = self.recurrent_delays.to(device=device, dtype=dtype)    # (N_in, N_out)

        if self.use_sig_p:
            s = 1.0 + 2.0 * self.sigma * torch.sigmoid(self.p_spread).to(device=device, dtype=dtype) # (N,)
            s_max = s.max()
            s = s.view(1, N, 1) # (N_out, 1)
        else:
            s = torch.tensor(1.0 + float(self.sigma), device=device, dtype=dtype)
            s_max = s
            
        L = int(torch.ceil(1.0 + torch.max(d) + s_max).item()) + 1
        support = torch.arange(L, device=device, dtype=dtype)              # (L,)

        mask = (torch.clamp(s - torch.abs(support[None, None, :] - (1.0 + d)[:, :, None]), min=0.0) / (s * s)) # (N_in, N_out, L)

        buffer = torch.zeros(B, N, L, device=device, dtype=dtype) # (B, N_in, L)
        pointer = 0

        if self.store_v_seq:
            v_seq = []

        for t in range(T):
            rec_now = buffer[:, :, pointer]                       # (B, N_in)

            y = self.neuron_module.single_step_forward(x_seq[t] + self.dropout(rec_now))  # (B, N_out)
    
            X_rec = torch.einsum('bo,io,iol->bil', y, W, mask) # (B, N_in, L)
            
            buffer[:, :, pointer] = 0.0
            pointer = (pointer + 1) % L

            first_chunk = min(L - 1, L - pointer)
            if first_chunk > 0:
                buffer[:, :, pointer:pointer + first_chunk].add_(X_rec[:, :, 1:1 + first_chunk])
            remaining = L - 1 - first_chunk
            if remaining > 0:
                buffer[:, :, 0:remaining].add_(X_rec[:, :, 1 + first_chunk:1 + first_chunk + remaining])

            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.neuron_module.v)

        self.x_seq = x_seq
        
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def forward(self, x_seq: torch.Tensor):
        if self.step_mode == 's':
            return super().single_step_forward(x_seq)
        else:
            return self.multi_step_forward(x_seq)
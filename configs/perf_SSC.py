from spikingjelly.activation_based import neuron, surrogate
from src.utils import Triangle

class Config():
    
    ### Dataset ###
    
    dataset = 'SSC'
    datasets_path = 'Datasets/SSC'
    seed = 0
    time_window = 250
    n_bins = 5
    
    ### General ###
    
    epochs = 100
    batch_size = 128
    
    bias = True
    use_batch_norm = False
    
    results_dir = ''
    
    ### Model architechture ###
    
    hidden_layers = [256, 256, 256]
    
    input_size = 700//n_bins
    output_size = 35
    
    recurrent_dropout_rate = 0.3
    feedforward_dropout_rate = 0.1
    
    init_ff_weights = 'default' # 'kaiming' or 'normal' or 'default'
    init_dcls_weights = 'default' # 'kaiming' or 'normal' or 'default'
    
    no_delay_in_first_layer = False
    no_delay_in_last_layer = False
    
    ### Spiking neuron configuration ###
    
    neuron_module = neuron.LIFNode
    backend = 'torch'
    
    tau = 2.0
    v_threshold = 1.0
    v_reset = None # None if soft reset
    
    surrogate_function = Triangle.apply
    detach_reset = False
    decay_input=False
    
    step_mode = 'm'
    store_v_seq = True
    
    ### Recurrent delays ###
    
    use_sig_p = True
    
    delay_std_init = 12 
    init_rec_delay = 'half_normal' # 'half_normal' or 'uniform'
    rec_delay_init_gain = 1.0
    
    sigma_init = 10.0
    sigma_decay = 0.95
    
    ### Feedforward delays ###
    
    DCLSversion = 'v1' # 'gauss' not implemented yet
    
    kernel_count = 1
    max_feedforward_delay = 25
    max_feedforward_delay = max_feedforward_delay if max_feedforward_delay%2==1 else max_feedforward_delay+1 
    
    left_padding = max_feedforward_delay - 1
    right_padding = 0
    
    init_pos_a = -max_feedforward_delay//2
    init_pos_b = max_feedforward_delay//2
    
    ### Optimization ###
    
    optim = 'adam'
    
    scheduler_weights = 'onecycle' # 'cos' or 'onecycle'
    scheduler_pos = 'onecycle' # 'onecycle' or 'cos'
    
    lr_w = 1e-3
    lr_positions = 5e-2
    
    weight_decay = 1e-5
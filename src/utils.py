import torch
import numpy as np
import random
import os
import sys
import time
from prettytable import PrettyTable

class Triangle(torch.autograd.Function):
    """Altered from code of Temporal Efficient Training, ICLR 2022 (https://openreview.net/forum?id=_XNtisL32jv)
    max(0, 1 − |ui[t] − θ|)"""

    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None
    
def reset_states(model):
    for m in model.modules():
        if hasattr(m, 'reset'):
            m.reset()
            
try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except:
    term_width = 80
            
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None, every=50):
    if every is None or every < 1:
        every = 1
    if (current not in (0, total - 1)) and (current % every != 0):
        return
    global _pb_disabled
    if '_pb_disabled' in globals() and _pb_disabled:
        return

    global last_time, begin_time

    # initialize timers even if the first printed step isn't 0
    if current == 0 or 'begin_time' not in globals():
        begin_time = time.time()
    if 'last_time' not in globals():
        last_time = time.time()

    try:
        cur_len  = int(TOTAL_BAR_LENGTH*current/total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for _ in range(cur_len):  sys.stdout.write('=')
        sys.stdout.write('>')
        for _ in range(rest_len): sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time  = time.time()
        step_time = cur_time - last_time
        last_time = cur_time
        tot_time  = cur_time - begin_time

        parts = [f'  Step: {format_time(step_time)}', f' | Tot: {format_time(tot_time)}']
        if msg: parts.append(' | ' + msg)
        line = ''.join(parts)

        sys.stdout.write(line)

        pad = max(0, term_width - int(TOTAL_BAR_LENGTH) - len(line) - 3)
        for _ in range(pad): sys.stdout.write(' ')

        for _ in range(term_width - int(TOTAL_BAR_LENGTH/2) + 2): sys.stdout.write('\b')
        sys.stdout.write(f' {current+1}/{total} ')

        if current < total - 1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')

        sys.stdout.flush()

    except OSError as e:
        if getattr(e, "errno", None) == 28:
            _pb_disabled = True
            return
        raise

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def seed_everything(seed=0, is_cuda=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if is_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def calc_loss_SHD(output, y):
    # (T, B, N)
    softmax_fn = torch.nn.Softmax(dim=2)
    m = torch.sum(softmax_fn(output), 0)
    CEloss = torch.nn.CrossEntropyLoss()
    loss = CEloss(m, y)
    return loss

def calc_metric_SHD(output, y):
    # (T, B, N)
    # mean accuracy over batch
    softmax_fn = torch.nn.Softmax(dim=2)
    m = torch.sum(softmax_fn(output), 0)
    return np.mean((torch.max(y,1)[1]==torch.max(m,1)[1]).detach().cpu().numpy())

def calc_loss_SSC(output, y):
    # (T, B, N)
    m = torch.sum(output, 0) # (B, N)
    CEloss = torch.nn.CrossEntropyLoss()
    loss = CEloss(m, y)
    return loss # (B)

def calc_metric_SSC(output, y):
    # (T, B, N)
    m = torch.sum(output, 0) # (B, N)
    _, predicted = m.max(1) # (B,)
    return predicted.eq(y).sum().item()
from helper import *
import torch.nn as nn
from math import sqrt, floor
import random
import time

def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # Compute loss for each (trial, timestep) (average accross output dimensions)
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()

def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          mask_gradients=False, clip_gradient=None, early_stop=None, keep_best=False, cuda=False, resample=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param mask_gradients: bool, set to True if training the SupportLowRankRNN_withMask for reduced models
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param early_stop: None or float, set to target loss value after which to immediately stop if attained
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :param resample: for SupportLowRankRNNs, set True
    :return: nothing
    """
    print("Training...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    if plot_gradient:
        gradient_norms = []

    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    # Initialize setup to keep best network
    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        if keep_best:
            best = net.clone()
            best_loss = initial_loss.item()

    # Training loop
    for epoch in range(n_epochs):
        begin = time.time()
        losses = []  # losses over the whole epoch
        for i in range(num_examples // batch_size):
            optimizer.zero_grad()
            random_batch_idx = random.sample(range(num_examples), batch_size)
            batch = input[random_batch_idx]
            output = net(batch)
            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
            losses.append(loss.item())
            all_losses.append(loss.item())
            loss.backward()
            if mask_gradients:
                net.m.grad = net.m.grad * net.m_mask
                net.n.grad = net.n.grad * net.n_mask
                net.wi.grad = net.wi.grad * net.wi_mask
                net.wo.grad = net.wo.grad * net.wo_mask
                net.unitn.grad = net.unitn.grad * net.unitn_mask
                net.unitm.grad = net.unitm.grad * net.unitm_mask
                net.unitwi.grad = net.unitwi.grad * net.unitwi_mask
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            if plot_gradient:
                tot = 0
                for param in [p for p in net.parameters() if p.requires_grad]:
                    tot += (param.grad ** 2).sum()
                gradient_norms.append(sqrt(tot))
            optimizer.step()
            # These 2 lines important to prevent memory leaks
            loss.detach_()
            output.detach_()
            if resample:
                net.resample_basis()
        if keep_best and np.mean(losses) < best_loss:
            best = net.clone()
            best_loss = np.mean(losses)
            print("epoch %d:  loss=%.3f  (took %.2f s) *" % (epoch, np.mean(losses), time.time() - begin))
        else:
            print("epoch %d:  loss=%.3f  (took %.2f s)" % (epoch, np.mean(losses), time.time() - begin))
        if early_stop is not None and np.mean(losses) < early_stop:
            break

    if plot_learning_curve:
        plt.plot(all_losses)
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.plot(gradient_norms)
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())

class FullRankRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha=0.2, rho=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False,
                 wi_init=None, wo_init=None, wrec_init=None, si_init=None, so_init=None):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float, value of dt/tau
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        """
        super(FullRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.non_linearity = torch.tanh

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if wrec_init is None:
                self.wrec.normal_(std=rho / sqrt(hidden_size))
            else:
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / hidden_size)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            self.h0.zero_()
        self.wi_full, self.wo_full = [None] * 2
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input, return_dynamics=False):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self._define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * \
                (-h + r.matmul(self.wrec.t()) + input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = FullRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                              self.rho, self.train_wi, self.train_wo, self.train_wrec, self.train_h0,
                              self.wi, self.wo, self.wrec, self.si, self.so)
        return new_net
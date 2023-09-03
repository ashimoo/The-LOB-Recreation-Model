import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint
import utils
from torch.nn.modules.rnn import RNNCell, GRUCell, LSTMCell
import torchode as to
import numpy as np

class ODEFunc(nn.Module):
	def __init__(self, ode_func_net):
		"""
		ode_func_net: neural net that used to transform hidden state in ode
		"""
		super(ODEFunc, self).__init__()
		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""
		Perform one step in solving ODE. Given current data point y and
		current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)

class DiffeqSolver(nn.Module):
	def __init__(self, ode_func, method, odeint_rtol=1e-4, odeint_atol=1e-5):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver.
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
			rtol = self.odeint_rtol, atol = self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)

		assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver using samples from the prior
		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        # input: (batch_size, time_length, num_features)

        query = self.query(input)
        key = self.key(input)
        value = self.value(input)

        attention_scores = torch.matmul(query, key.transpose(1, 2))
        attention_weights = self.softmax(attention_scores)

        attended_values = torch.matmul(attention_weights, value)

        return attended_values[:,-1,:]

class Encoder_ODE_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, diffeq_solver, GRU_update=None,
                 n_gru_units = 64, device = torch.device("cuda")):
        super(Encoder_ODE_RNN, self).__init__()

        self.GRU_update = GRU_unit(latent_dim, input_dim, n_units=n_gru_units).to(device)
        self.diffeq_solver = diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

    def forward(self, data, time_steps, decoder, device='cuda'):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        n_traj, n_tp, n_dim = data.size()

        minimum_step = 1
        device = utils.get_device(data)
        data = torch.cat((data,torch.zeros((n_traj,1,n_dim)).to(device)),dim=1)
        prev_y = torch.zeros((n_traj, self.latent_dim)).to(device)

        t_eval_total, inverse_indices = torch.unique(time_steps, return_inverse=True)
        fetched_count = torch.zeros(n_traj,).to(device)
        mask = torch.zeros(n_traj,len(t_eval_total)).to(device)

        volume_dt = []
        time_intervals = []

        time_points_iter = range(0, len(t_eval_total))
        prev_t, t_i = -1, 0

        for i in time_points_iter:
            time_points = torch.linspace(prev_t, t_i, int((t_i - prev_t) / minimum_step + 1)).to(device)
            ode_sol = self.diffeq_solver(prev_y.unsqueeze(0), time_points)[0]
            volume_dt.append(torch.mean(decoder(ode_sol[:,:-1]),1))
            update_mask = (time_steps == t_i).any(dim=1, keepdim=True).float()
            fetched_x = data[list(range(n_traj)),fetched_count.long()] * update_mask
            yi = self.GRU_update(ode_sol[:,-1].unsqueeze(0), fetched_x.unsqueeze(0))

            mask[:,i] = (time_steps[:,-1]>=t_i)
            fetched_count = fetched_count + update_mask.squeeze(1)
            if i < time_points_iter[-1]:
                prev_t, t_i = t_eval_total[i], t_eval_total[i+1]
            prev_y = yi.squeeze(0) * update_mask + ode_sol[:,-1] * (1-update_mask)

        volume_dt = torch.stack(volume_dt).permute(1,0,2)
        selected_state = torch.take_along_dim(volume_dt[:,1:], inverse_indices[:,:-1,None],dim=1)
        return prev_y, selected_state, (time_steps[:,1:] - time_steps[:,:-1])

class Simple_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, n_gru_units = 64, device = torch.device("cuda")):
        super(Simple_RNN, self).__init__()

        self.GRU_update = GRU_unit(latent_dim, input_dim, n_units=n_gru_units).to(device)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

    def forward(self, data, time_steps, device='cuda'):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        n_traj, n_tp, _ = data.size()

        prev_y = torch.zeros((n_traj, self.latent_dim)).to(device)
        prev_t, t_i = time_steps[:,0] - 1, time_steps[:,0]

        latent_ys = []
        time_intervals = []

        time_points_iter = range(0, len(time_steps[0]))

        for i in time_points_iter:

            xi = data[:, i, :].unsqueeze(0)
            yi = self.GRU_update(prev_y.unsqueeze(0), xi)
            prev_y = yi.squeeze(0)
            if i < time_points_iter[-1]:
                prev_t, t_i = time_steps[:,i], time_steps[:,i + 1]
            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, 1).squeeze(0).permute(1,0,2)
        time_intervals = time_steps[:,1:]-time_steps[:,:-1]

        return yi, latent_ys[:,:-1,:], time_intervals

class RNN_decay(nn.Module):
    def __init__(self, latent_dim, input_dim, GRU_update=None,
                 n_gru_units = 64, device = torch.device("cuda"), time = False):
        super(RNN_decay, self).__init__()

        if time:
            self.GRU_update = GRU_unit_(latent_dim, input_dim + 1, n_units=n_gru_units).to(device)
        else:
            self.GRU_update = GRU_unit_(latent_dim, input_dim, n_units=n_gru_units).to(device)

        self.decay_layer = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,1),
            nn.ReLU())
        utils.init_network_weights(self.decay_layer)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

    def forward(self, data, time_steps, decoder, device='cuda'):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        n_traj, n_tp, n_dims = data.size()
        device = utils.get_device(data)

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_y_target = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_t, t_i = time_steps[:,0] - 1, time_steps[:,0]

        volume_dt = []
        time_intervals = []

        time_points_iter = range(0, len(time_steps[0]))
        interpolate = 2
        for i in time_points_iter:
            dt = t_i - prev_t
            dt = dt[:,None] * torch.linspace(0,1,interpolate+1).to(device)[None,:]

            if i > 0:
                decay_coef = self.decay_layer(prev_y)
                yi_decay_ = prev_y_target.unsqueeze(-2) + (prev_y - prev_y_target).unsqueeze(-2) * torch.exp(-decay_coef * dt.unsqueeze(0)).unsqueeze(-1)
                yi_decay = yi_decay_[:,:,-1,:]
                volume_dt.append(torch.mean(decoder(yi_decay_[0,:,:-1,:]),dim=-2))
            else:
                yi_decay = prev_y

            xi = data[:, i, :].unsqueeze(0)
            prev_y,prev_y_target = self.GRU_update(yi_decay,prev_y_target,xi)
            if i < time_points_iter[-1]:
                prev_t, t_i = time_steps[:,i], time_steps[:,i + 1]
        volume_dt = torch.stack(volume_dt).permute(1,0,2)
        time_intervals = time_steps[:,1:]-time_steps[:,:-1]

        return prev_y, volume_dt, time_intervals

class Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim, device, n_labels=4):
        super(Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.device = device

    def compute_all_losses(self, batch_dict, side, time = False):
        info = self.get_reconstruction(batch_dict["data"], batch_dict["time_steps"], batch_dict["mask"], side, time)

        device = utils.get_device(batch_dict["data"])
        info['label_predictions'] = info['label_predictions'].unsqueeze(1)

        l1_loss = utils.compute_l1_loss(info["label_predictions"],batch_dict["labels"])

        if torch.isnan(l1_loss):
            print("label pred")
            print(info["label_predictions"])
            print("labels")
            print(batch_dict["labels"])
            raise Exception("l1 loss is Nan!")

        results = {}
        results["l1_loss"] = l1_loss

        return results


class GRU_unit_(nn.Module):
    def __init__(self, latent_dim, input_dim,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=64):
        super(GRU_unit_, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        self.update_gate_target = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        utils.init_network_weights(self.update_gate)
        utils.init_network_weights(self.update_gate_target)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        self.reset_gate_target = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        utils.init_network_weights(self.reset_gate)
        utils.init_network_weights(self.reset_gate_target)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim))
        utils.init_network_weights(self.new_state_net)


    def forward(self, y_mean, y_mean_target, x):  # y is actually h
        input_dim = x.size()[-1]
        y_concat = torch.cat([y_mean, x[:,:,:int(input_dim/2)]], -1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        update_gate_target = self.update_gate_target(y_concat)
        reset_gate_target = self.reset_gate_target(y_concat)
        concat = torch.cat([y_mean * reset_gate, x[:,:,:int(input_dim/2)]], -1)
        new_state= self.new_state_net(concat)  # h'
        new_y = (1 - update_gate) * new_state + update_gate * y_mean  # new h
        new_y_target = (1 - update_gate_target) * new_state + update_gate_target * y_mean_target

        mask = (torch.sum(x[:,:,int(input_dim/2):], -1, keepdim=True) > 0).float()
        new_y = mask * new_y + (1 - mask) * y_mean
        new_y_target = mask * new_y_target + (1 - mask) * y_mean_target

        return new_y,new_y_target  # y has the dimension of latent_dim

class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=64):
        super(GRU_unit, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())

        utils.init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        utils.init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim))
        utils.init_network_weights(self.new_state_net)


    def forward(self, y_mean, x):  # y is actually h
        input_dim = x.size()[-1]
        y_concat = torch.cat([y_mean, x[:,:,:int(input_dim/2)]], -1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, x[:,:,:int(input_dim/2)]], -1)
        new_state= self.new_state_net(concat)  # h'
        new_y = (1 - update_gate) * new_state + update_gate * y_mean  # new h

        mask = (torch.sum(x[:,:,int(input_dim/2):], -1, keepdim=True) > 0).float()
        new_y = mask * new_y + (1 - mask) * y_mean
        return new_y

class LOBRM(Baseline):
    def __init__(self, main_module, input_dim, WS = True, HC = True, ES = True, device=torch.device("cuda"),
                 diffeq_solver=None, size_latent = 32, size_latent_WS = 16, time = False, n_gru_units=64, n_units=64, n_labels=4):

        Baseline.__init__(self, input_dim, size_latent, device=device, n_labels=n_labels)
        self.module_name = main_module
        self.WS = WS
        self.HC = HC
        self.ES = ES
        self.n_labels = n_labels

        if self.ES:
            if main_module == 'decay':
                self.main_module = RNN_decay(
                    latent_dim=size_latent,
                    input_dim=(input_dim),
                    n_gru_units=n_gru_units,
                    device=device,
                    time = time).to(device)
            elif main_module == 'ode':
                self.main_module = Encoder_ODE_RNN(
                    latent_dim=size_latent,
                    input_dim=(input_dim),
                    diffeq_solver=diffeq_solver,
                    n_gru_units=n_gru_units,
                    device=device).to(device)
            elif main_module == 'attention':
                self.main_module = SelfAttention((input_dim),size_latent).to(device)
            elif main_module == 'simple':
                self.main_module = Simple_RNN(
                    latent_dim=size_latent,
                    input_dim=(input_dim)).to(device)

            self.decoder_increment = nn.Sequential(
                nn.Linear(size_latent,n_units),
                nn.Tanh(),
                nn.Linear(n_units, n_units),
                nn.Tanh(),
                nn.Linear(n_units,n_labels))
            utils.init_network_weights(self.decoder_increment)

        if self.WS:
            self.encoder_weight = nn.GRU(n_labels,size_latent_WS,batch_first=True)
            self.decoder_weight = nn.Sequential(
                nn.Linear(size_latent_WS, n_labels),
                nn.Sigmoid())
            utils.init_network_weights(self.encoder_weight)
            utils.init_network_weights(self.decoder_weight)

        if self.HC:
            self.decoder_base = nn.Sequential(
                nn.Linear(size_latent,n_units),
                nn.Tanh(),
                nn.Linear(n_units,n_units),
                nn.Tanh(),
                nn.Linear(n_units, n_labels),
            )
            utils.init_network_weights(self.decoder_base)
            self.rnn_cell_base = GRUCell(n_labels, size_latent)
            utils.init_network_weights(self.rnn_cell_base)

    def get_reconstruction(self, data, time_steps, mask, side, time = False):

        n_traj, n_tp, n_dims = data.size()
        data_and_mask = torch.cat([data, mask], -1)
        extra_info = {}
        if self.ES:

            if self.module_name == 'ode':
                _, latent_state, time_intervals = self.main_module(data_and_mask, time_steps, self.decoder_increment)
                increment_pred = torch.sum(latent_state * time_intervals[:,:,None],1) / time_steps[:,-1].unsqueeze(-1)

            if self.module_name == 'decay':
                _, volume_dt, time_intervals = self.main_module(data_and_mask, time_steps, self.decoder_increment)
                increment_pred = torch.sum(volume_dt,1) / torch.sum(time_intervals,1,keepdim=True)

            elif self.module_name == 'simple':
                _, latent_ys, time_intervals = self.main_module(data_and_mask, time_steps)

                outputs = self.decoder_increment(latent_ys)
                increment_pred = torch.sum(outputs * time_intervals.unsqueeze(-1),1) / torch.sum(time_intervals,1).unsqueeze(-1)

            elif self.module_name == 'attention':
                context_vec = self.main_module(data)
                increment_pred = self.decoder_increment(context_vec)

        if self.HC:
            if side == "ask":
                data_for_base = data[:, :, 8:12]
                mask_for_base = mask[:, :, 8:12]
            elif side == "bid":
                data_for_base = data[:, :, 18:22]
                mask_for_base = mask[:, :, 18:22]
            hidden_state_base, _ = rnn_time(data_for_base , embedding = None, cell=self.rnn_cell_base, mask=mask_for_base)
            base_pred = self.decoder_base(hidden_state_base.squeeze(0))

        if self.WS:
            _, mask_encoded = self.encoder_weight(mask_for_base)
            weights_1 = self.decoder_weight(mask_encoded.squeeze(0))
            device = utils.get_device(weights_1)
            ones = torch.ones(len(mask_for_base), self.n_labels).to(device)
            weights_2 = ones - weights_1

        extra_info = {}
        if self.HC and self.ES and self.WS:
            extra_info["label_predictions"] = weights_1 * increment_pred + weights_2 * base_pred
        elif self.HC and self.ES and not self.WS:
            extra_info["label_predictions"] = 0.5 * base_pred + 0.5 * increment_pred
        elif self.HC and not self.ES and not self.WS:
            extra_info["label_predictions"] = base_pred
        elif not self.HC and self.ES and not self.WS:
            extra_info["label_predictions"] = increment_pred
        else:
            raise Exception("wrong combination of modules")

        return extra_info

def rnn_time(inputs, embedding, cell, delta_ts = None, mask = None, n_steps=0, masked_update = True):
    if n_steps == 0:
        n_steps = inputs.size(1)
    n_dim = inputs.size(2)
    all_hiddens = []
    hidden_state = None
    cell_state = None

    for i in range(n_steps):
        if n_dim == 100:
            if embedding:
                rnn_input = embedding(inputs[:,i])
            else:
                rnn_input = inputs[:,i]
        else:
            rnn_input = inputs[:,i]
        if mask is not None:
            mask_i = mask[:,i,:]
        if delta_ts is not None:
            delta_t = delta_ts[:, i].unsqueeze(1)
            rnn_input = torch.cat((rnn_input, delta_t), -1).squeeze(1)

        prev_hidden = hidden_state
        if not isinstance(cell, LSTMCell):
            hidden_state = cell(rnn_input, hidden_state)
        else:
            if i == 0:
                hidden_state , cell_state = cell(rnn_input)
            else:
                hidden_state, cell_state = cell(rnn_input,(hidden_state,cell_state))

        if masked_update and (mask is not None) and (prev_hidden is not None):
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            summed_mask = (torch.sum(mask_i, -1, keepdim = True) > 0).float()
            assert(not torch.isnan(summed_mask).any())
            hidden_state = summed_mask * hidden_state + (1-summed_mask) * prev_hidden

        all_hiddens.append(hidden_state)

    all_hiddens = torch.stack(all_hiddens, 0)
    all_hiddens = all_hiddens.permute(1,0,2).unsqueeze(0)
    return hidden_state, all_hiddens


class OdeNet(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(OdeNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden))

    def forward(self, y):
        return self.layers(y)


def create_mask_matrix(A):
    n = torch.max(A).item() + 1
    m = A.size(0)
    return torch.arange(n).expand(m, n) < A.unsqueeze(1)

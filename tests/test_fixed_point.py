import torch
from torch import nn
from mat6115.fixed_point import FixedPointFinder


def test_rnn_untouched():
    """
    This test is to make sure the weights of the RNN are not updated.
    Only the provided hidden state should change.
    """
    rnn = nn.GRU(input_size=5, hidden_size=11, batch_first=True)
    bias_hh_l0 = rnn.bias_hh_l0.clone()
    bias_ih_l0 = rnn.bias_ih_l0.clone()
    weight_hh_l0 = rnn.weight_hh_l0.clone()
    weight_ih_l0 = rnn.weight_ih_l0.clone()

    constant_input = torch.zeros((20, 1, 5))
    hidden_state = nn.init.normal_(torch.empty((1, 20, 11)))
    initial_hidden = hidden_state.clone()

    fixed_point_finder = FixedPointFinder(
        rnn_cell=rnn, n_iter=10, lr=0.01, batch_size=3
    )
    point, is_fixed_point = fixed_point_finder.run(hidden_state, constant_input)
    assert len(point) == constant_input.shape[0]
    assert (bias_hh_l0 == rnn.bias_hh_l0).all()
    assert (bias_ih_l0 == rnn.bias_ih_l0).all()
    assert (weight_hh_l0 == rnn.weight_hh_l0).all()
    assert (weight_ih_l0 == rnn.weight_ih_l0).all()
    assert (initial_hidden != hidden_state).all()
    assert (constant_input == 0).all()


def test_calc_jacobian():
    """
    Not a very scientific test. Basically just checking that everything runs.
    """
    rnn = nn.GRU(input_size=5, hidden_size=11, batch_first=True)

    constant_input = torch.zeros((1, 1, 5))
    hidden_state = nn.init.normal_(torch.empty((1, 1, 11)))

    fixed_point_finder = FixedPointFinder(
        rnn_cell=rnn, n_iter=10, lr=0.01, batch_size=3
    )

    jacobian_h, jacobian_i = fixed_point_finder.calc_jacobian(
        hidden_state, constant_input
    )
    assert jacobian_h.shape == (11, 11)
    assert jacobian_i.shape == (11, 5)

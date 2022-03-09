import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from allocationmechanisms import PartialAllocationFS
from preferencemodels import LinearFehrSchmidt

if __name__ == '__main__':
    do_save_ = True
    save_path_ = os.path.join('..', 'results', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    # ----- Choose FS params -----
    a_mat_ = np.array([
        [1., 0., 2.],
        [0., 1., 1.]
    ])

    b_ = np.array([
        1,
        1.
    ])

    n_bid_ = a_mat_.shape[0]

    # ----- Allocate for different levels if IA -----
    alphas_1_ = np.arange(0, 0.5, 5e-3)
    alphas_2_ = np.zeros((len(alphas_1_),))

    alphas_ = np.concatenate((alphas_1_.reshape((1, -1)), alphas_2_.reshape((1, -1))), axis=0)

    n_test_ = alphas_.shape[1]

    # ----- Loop over IAs -----
    utilities_pa_ = np.zeros((n_bid_, n_test_))
    utilities_pa_bar_ = np.zeros((n_bid_, n_test_))
    valuations_pa_ = np.zeros((n_bid_, n_test_))
    valuations_pa_bar_ = np.zeros((n_bid_, n_test_))
    for idx_ in tqdm(range(n_test_)):
        alpha_ = alphas_[:, idx_]

        # Allocate
        pa_ = PartialAllocationFS(a_mat_, alpha_, alpha_, b_)
        _, x_pa_mat_ = pa_.allocate()

        alpha_bar_ = alpha_.copy()
        alpha_bar_[0] = 0
        pa_bar_ = PartialAllocationFS(a_mat_, alpha_bar_, alpha_bar_, b_)
        _, x_pa_bar_mat_ = pa_bar_.allocate()

        # Find allocation efficiency
        utilities_pa_[:, idx_] = np.sum(x_pa_mat_*a_mat_, axis=1)
        utilities_pa_bar_[:, idx_] = np.sum(x_pa_bar_mat_*a_mat_, axis=1)

        pref_ = LinearFehrSchmidt(a_mat_, alpha_, alpha_, utilities_pa_[:, idx_])
        valuations_pa_[:, idx_] = pref_.v(x_pa_mat_)

        pref_bar_ = LinearFehrSchmidt(a_mat_, alpha_, alpha_, utilities_pa_bar_[:, idx_])
        valuations_pa_bar_[:, idx_] = pref_bar_.v(x_pa_bar_mat_)

    # ----- Plot -----
    plt.figure(figsize=(4, 4))

    plt.plot(alphas_[0], valuations_pa_[0], 'b')
    plt.plot(alphas_[0], valuations_pa_bar_[0], 'b--')
    plt.plot(alphas_[0], valuations_pa_[1], 'r')
    plt.plot(alphas_[0], valuations_pa_bar_[1], 'r--')
    plt.legend([r'$v_1^\alpha(x^{PA})$', r'$v_1^\alpha(\bar{x}^{PA})$',
                r'$v_2^\alpha(x^{PA})$', r'$v_2^\alpha(\bar{x}^{PA})$'])
    plt.title(r'$x^{PA}=PA((v_1^\alpha, v_2^0)), \; \bar{x}^{PA}=PA((v_1^0, v_2^0))$')
    plt.xlabel(r'$\alpha$')

    if do_save_:
        plt.savefig(os.path.join(save_path_, 'monotonicity_of_pa.pdf'))

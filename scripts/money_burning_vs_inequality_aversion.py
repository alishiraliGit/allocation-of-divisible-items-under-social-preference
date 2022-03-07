import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from allocationmechanisms import ProportionallyFairFS, PartialAllocationFS
from preferencemodels import LinearFehrSchmidt

if __name__ == '__main__':
    do_save_ = False
    save_path_ = os.path.join('..', 'results', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    # ----- Choose FS params -----
    a_mat_ = np.array([
        [0.33, 0.79, 0.30],
        [0.45, 0.13, 0.40]
    ])

    b_ = np.array([
        1,
        1.
    ])

    n_bid_ = a_mat_.shape[0]

    # ----- Allocate for different levels if IA -----
    alphas_ = np.tile(np.arange(0, 0.05, 5e-3).reshape((1, -1)), reps=(n_bid_, 1))
    betas_ = alphas_

    n_test_ = alphas_.shape[1]

    # ----- Loop over IAs -----
    approx_factors_ = np.zeros((n_test_,))
    money_burnings_ = np.zeros((n_test_,))
    utilities_pa_ = np.zeros((n_bid_, n_test_))
    utilities_pf_ = np.zeros((n_bid_, n_test_))
    valuations_pa_ = np.zeros((n_bid_, n_test_))
    valuations_pf_ = np.zeros((n_bid_, n_test_))
    for idx_ in tqdm(range(n_test_)):
        alpha_ = alphas_[:, idx_]
        beta_ = betas_[:, idx_]

        # Allocate
        pf_ = ProportionallyFairFS(a_mat_, alpha_, beta_, b_)
        pa_ = PartialAllocationFS(a_mat_, alpha_, beta_, b_)

        _, x_star_mat_, _ = pf_.allocate()
        _, x_pa_mat_ = pa_.allocate()

        # Find allocation efficiency
        approx_factors_[idx_] = np.min(x_pa_mat_/x_star_mat_)
        money_burnings_[idx_] = 1 - np.sum(x_pa_mat_)/np.sum(x_star_mat_)

        utilities_pa_[:, idx_] = np.sum(x_pa_mat_*a_mat_, axis=1)
        utilities_pf_[:, idx_] = np.sum(x_star_mat_*a_mat_, axis=1)

        pref_ = LinearFehrSchmidt(a_mat_, alpha_, beta_, utilities_pa_[:, idx_])
        valuations_pa_[:, idx_] = pref_.v(x_pa_mat_)

        pref_ = LinearFehrSchmidt(a_mat_, alpha_, beta_, utilities_pf_[:, idx_])
        valuations_pf_[:, idx_] = pref_.v(x_star_mat_)

    # ----- Plot -----
    plt.figure(figsize=(4, 4))
    plt.plot(alphas_[0], approx_factors_, 'k')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('approx. factor')
    plt.tight_layout()

    if do_save_:
        plt.savefig(os.path.join(save_path_, 'approx_factor_vs_alpha_for_two_bidders.pdf'))

    plt.figure(figsize=(4, 4))
    plt.plot(alphas_[0], money_burnings_, 'k')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('money burned')
    plt.tight_layout()

    if do_save_:
        plt.savefig(os.path.join(save_path_, 'money_burned_vs_alpha_for_two_bidders.pdf'))

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(alphas_[0], valuations_pf_[0], 'b')
    plt.plot(alphas_[0], utilities_pf_[0], 'b--')
    plt.plot(alphas_[0], valuations_pf_[1], 'r')
    plt.plot(alphas_[0], utilities_pf_[1], 'r--')
    plt.legend([r'$v_1^*$', r'$u_1^*$', r'$v_2^*$', r'$u_2^*$'])
    plt.title('u and v for PF')
    plt.xlabel(r'$\alpha$')

    plt.subplot(1, 2, 2)
    plt.plot(alphas_[0], valuations_pa_[0], 'b')
    plt.plot(alphas_[0], utilities_pa_[0], 'b--')
    plt.plot(alphas_[0], valuations_pa_[1], 'r')
    plt.plot(alphas_[0], utilities_pa_[1], 'r--')
    plt.title('u and v for PA')
    plt.legend([r'$v_1^{pa}$', r'$u_1^{pa}$', r'$v_2^{pa}$', r'$u_2^{pa}$'])
    plt.xlabel(r'$\alpha$')

    if do_save_:
        plt.savefig(os.path.join(save_path_, 'u_v_for_two_bidders.pdf'))

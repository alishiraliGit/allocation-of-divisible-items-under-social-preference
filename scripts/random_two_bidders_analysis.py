import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from allocationmechanisms import ProportionallyFairFS, PartialAllocationFS
from preferencemodels import LinearFehrSchmidt

rng = np.random.default_rng(1)

if __name__ == '__main__':
    # ----- Settings -----
    # General
    n_bid_ = 2
    n_item_ = 3

    n_rept_ = 10
    do_save_ = False

    # Allocation mech.
    b_ = np.array([
        1,
        1.
    ])

    # Preference
    alphas_ = np.tile(np.arange(0, 0.2, 1e-2).reshape((1, -1)), reps=(n_bid_, 1))
    betas_ = alphas_

    n_ia_ = alphas_.shape[1]

    # Path
    save_path_ = os.path.join('..', 'results', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    # ----- Big loop -----
    # Random interests
    a_mats_ = rng.random((n_rept_, n_bid_, n_item_))
    # Init.
    approx_factors_ = np.zeros((n_rept_, n_ia_,))
    money_burnings_ = np.zeros((n_rept_, n_ia_,))
    utilities_pa_ = np.zeros((n_rept_, n_bid_, n_ia_))
    utilities_pf_ = np.zeros((n_rept_, n_bid_, n_ia_))
    valuations_pa_ = np.zeros((n_rept_, n_bid_, n_ia_))
    valuations_pf_ = np.zeros((n_rept_, n_bid_, n_ia_))
    r_ = np.zeros((n_rept_,))

    for i_rept_ in tqdm(range(n_rept_)):
        a_mat_ = rng.random((n_bid_, n_item_))

        # Loop over IAs
        for ia_idx_ in range(n_ia_):
            alpha_ = alphas_[:, ia_idx_]
            beta_ = betas_[:, ia_idx_]

            # Allocate
            pf_ = ProportionallyFairFS(a_mat_, alpha_, beta_, b_)
            pa_ = PartialAllocationFS(a_mat_, alpha_, beta_, b_)

            _, x_star_mat_, _ = pf_.allocate()
            _, x_pa_mat_ = pa_.allocate()

            # Find allocation efficiency
            approx_factors_[i_rept_, ia_idx_] = np.min(x_pa_mat_/x_star_mat_)
            money_burnings_[i_rept_, ia_idx_] = 1 - np.sum(x_pa_mat_)/np.sum(x_star_mat_)

            utilities_pa_[i_rept_, :, ia_idx_] = np.sum(x_pa_mat_*a_mat_, axis=1)
            utilities_pf_[i_rept_, :, ia_idx_] = np.sum(x_star_mat_*a_mat_, axis=1)

            pref_ = LinearFehrSchmidt(a_mat_, alpha_, beta_, utilities_pa_[i_rept_, :, ia_idx_])
            valuations_pa_[i_rept_, :, ia_idx_] = pref_.v(x_pa_mat_)
            valuations_pf_[i_rept_, :, ia_idx_] = pref_.v(x_star_mat_)

        # Find corr coefficient
        r_[i_rept_] = np.corrcoef(a_mat_)[0, 1]

    # ----- Plot -----
    plt.figure(figsize=(4, 4))
    plt.plot(alphas_[0], approx_factors_.T, 'k', alpha=0.4)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('approx. factor')
    plt.tight_layout()

    if do_save_:
        plt.savefig(os.path.join(save_path_, 'approx_factor_vs_alpha_for_random_bidders.pdf'))

    plt.figure(figsize=(4, 4))
    plt.plot(alphas_[0], money_burnings_.T, 'k', alpha=0.4)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('money burned')
    plt.tight_layout()

    if do_save_:
        plt.savefig(os.path.join(save_path_, 'money_burned_vs_alpha_for_random_bidders.pdf'))

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(alphas_[0], valuations_pf_[:, 0, :].T, 'b', alpha=0.4)
    plt.plot(alphas_[0], utilities_pf_[:, 0, :].T, 'b--', alpha=0.4)
    plt.plot(alphas_[0], valuations_pf_[:, 1, :].T, 'r', alpha=0.4)
    plt.plot(alphas_[0], utilities_pf_[:, 1, :].T, 'r--', alpha=0.4)
    plt.title('u and v for PF')
    plt.xlabel(r'$\alpha$')

    plt.subplot(1, 2, 2)
    plt.plot(alphas_[0], valuations_pa_[:, 0, :].T, 'b', alpha=0.4)
    plt.plot(alphas_[0], utilities_pa_[:, 0, :].T, 'b--', alpha=0.4)
    plt.plot(alphas_[0], valuations_pa_[:, 1, :].T, 'r', alpha=0.4)
    plt.plot(alphas_[0], utilities_pa_[:, 1, :].T, 'r--', alpha=0.4)
    plt.title('u and v for PA')
    plt.xlabel(r'$\alpha$')

    if do_save_:
        plt.savefig(os.path.join(save_path_, 'u_v_for_random_bidders.pdf'))

    plt.figure(figsize=(4, 4))
    plt.plot(r_, money_burnings_[:, 0], 'ro', alpha=0.4)
    plt.plot(r_, money_burnings_[:, -1], 'bo', alpha=0.4)
    plt.xlabel(r'$\rho$')
    plt.ylabel('money burned')
    plt.tight_layout()

    if do_save_:
        plt.savefig(os.path.join(save_path_, 'money_burned_vs_corrcoef_for_random_bidders.pdf'))

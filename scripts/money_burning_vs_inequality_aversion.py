import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from allocationmechanisms import ProportionallyFairFS, PartialAllocationFS

if __name__ == '__main__':
    do_save_ = False
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

    # ----- Allocate for different levels if IA -----
    alphas_ = np.tile(np.arange(0, 0.2, 5e-3).reshape((1, -1)), reps=(2, 1))
    betas_ = alphas_

    approx_factors_ = []
    money_burnings_ = []
    for idx_ in tqdm(range(alphas_.shape[1])):
        alpha_ = alphas_[:, idx_]
        beta_ = betas_[:, idx_]

        # Allocate
        pf_ = ProportionallyFairFS(a_mat_, alpha_, beta_, b_)
        pa_ = PartialAllocationFS(a_mat_, alpha_, beta_, b_)

        _, x_star_mat_, _ = pf_.allocate()
        _, x_pa_mat_ = pa_.allocate()

        # Find allocation efficiency
        approx_factors_.append(np.min(x_pa_mat_/x_star_mat_))
        money_burnings_.append(1 - np.sum(x_pa_mat_)/np.sum(x_star_mat_))

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

import numpy as np

from allocationmechanisms import ProportionallyFairFS, PartialAllocationFS

if __name__ == '__main__':
    # ----- Choose FS params -----
    a_mat_ = np.array([
        [1., 0., 2.],
        [0., 1., 1.]
    ])

    alpha_ = np.array([
        0.2,
        0.
    ])

    beta_ = np.array([
        0.2,
        0.
    ])

    b_ = np.array([
        1,
        1.
    ])

    # ----- Init. mechanisms -----
    pf_ = ProportionallyFairFS(a_mat_, alpha_, beta_, b_)
    pa_ = PartialAllocationFS(a_mat_, alpha_, beta_, b_)

    # ----- Allocate -----
    # PF
    success_, x_opt_, loss_opt_ = pf_.allocate(verbose=-1)

    print('PF:')
    print(success_)
    print(np.round(x_opt_, decimals=3))
    print(loss_opt_)

    # PA
    f_, x_sub_opt_ = pa_.allocate()

    print('PA:')
    print(f_)
    print(np.round(x_sub_opt_, decimals=3))

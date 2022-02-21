import numpy as np
from scipy.optimize import LinearConstraint, Bounds, minimize
from itertools import permutations

from preferencemodels import PreferenceModel, LinearFehrSchmidt


class ProportionallyFair:
    def __init__(self, pref_mdl: PreferenceModel, b):
        self.pref_mdl = pref_mdl
        self.n_bid, self.n_item = self.pref_mdl.n_bid, self.pref_mdl.n_item
        self.b = b

    def loss(self, x):
        v = self.pref_mdl.v(x)

        return -np.sum(self.b*np.log(v))

    def grad_loss(self, x):
        v = self.pref_mdl.v(x)
        grad_v = self.pref_mdl.grad_v(x)

        return -(self.b/v).reshape((1, -1)).dot(grad_v)[0]

    def constraints(self):
        n_bid, n_item = self.n_bid, self.n_item

        # Allocations per item to be <= 1
        c_per_item_mat = np.zeros((n_item, n_bid*n_item))
        for it in range(n_item):
            c_i = np.zeros((n_bid, n_item))
            c_i[:, it] = 1
            c_per_item_mat[it] = c_i.reshape((-1,), order='C')

        lin_const_per_item = LinearConstraint(
            A=c_per_item_mat,
            lb=-np.inf * np.ones((n_item,)),
            ub=np.ones((n_item,))
        )

        return [lin_const_per_item] + self.pref_mdl.constraints()

    def allocate(self, verbose=-1):
        n_bid, n_item = self.n_bid, self.n_item

        # Allocations to be >= 0
        bounds = Bounds([0]*n_bid*n_item, [np.inf]*n_bid*n_item)

        # Run the minimization algorithm
        res = minimize(
            fun=self.loss,
            x0=np.ones((n_bid*n_item,))/n_bid,
            method='trust-constr',
            jac=self.grad_loss,
            bounds=bounds,
            constraints=self.constraints(),
            options={'disp': True, 'verbose': verbose}
        )

        return res.success, res.x.reshape((n_bid, n_item), order='C'), res.fun


class ProportionallyFairFS:
    def __init__(self, a_mat, alpha, beta, b):
        self.a_mat = a_mat
        self.alpha = alpha
        self.beta = beta
        self.b = b

        self.n_bid, self.n_item = self.a_mat.shape

    def allocate(self, verbose=-1):
        n_bid, n_item = self.a_mat.shape

        opt_fun = np.inf
        opt_x = None
        for ord_u in permutations(range(n_bid)):
            pref_mdl = LinearFehrSchmidt(self.a_mat, self.alpha, self.beta, np.array(ord_u))
            pf = ProportionallyFair(pref_mdl, self.b)

            success, x_mat, loss = pf.allocate(verbose=verbose)

            if not success:
                return False, opt_x, opt_fun

            if loss < opt_fun:
                opt_fun = loss
                opt_x = x_mat

        return True, opt_x, opt_fun


class PartialAllocationFS:
    def __init__(self, a_mat, alpha, beta, b):
        self.a_mat = a_mat
        self.alpha = alpha
        self.beta = beta
        self.b = b

        self.n_bid, self.n_item = self.a_mat.shape

    def allocate_sub(self, exclude_bidders):
        # Drop the bidder
        bidders = np.array(range(self.n_bid))
        bidders_sub = np.delete(bidders, exclude_bidders)

        # Find PF allocation of the subset
        pf_fs = ProportionallyFairFS(
            self.a_mat[bidders_sub], self.alpha[bidders_sub], self.beta[bidders_sub], self.b[bidders_sub]
        )
        success, x_opt_mat, loss_opt = pf_fs.allocate(verbose=-1)

        # Find v of the optimum allocation
        u_opt = np.sum(self.a_mat[bidders_sub]*x_opt_mat, axis=1)
        fs_opt = LinearFehrSchmidt(
            self.a_mat[bidders_sub], self.alpha[bidders_sub], self.beta[bidders_sub], u_opt
        )
        v_opt = fs_opt.v(x_opt_mat.reshape((-1,), order='C'))

        return success, x_opt_mat, loss_opt, v_opt

    def allocate(self):
        # Find the PF including all bidders
        _, x_star_mat, _, v_star = self.allocate_sub([])

        # Find the fraction for each bidder
        f = np.zeros((self.n_bid,))
        bidders = np.array(range(self.n_bid))
        for i_bid in range(self.n_bid):
            _, x_mi_mat, _, v_mi_star = self.allocate_sub(i_bid)

            bidders_sub = np.delete(bidders, i_bid)
            f[i_bid] = (np.prod(v_star[bidders_sub]**self.b[bidders_sub]) /
                        np.prod(v_mi_star**self.b[bidders_sub]))**(1/self.b[i_bid])

        return f, x_star_mat*f.reshape((-1, 1))




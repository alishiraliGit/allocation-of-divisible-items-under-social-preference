import abc
import numpy as np
from scipy.optimize import LinearConstraint


class PreferenceModel(abc.ABC):
    def __init__(self, n_bid, n_item):
        self.n_bid = n_bid
        self.n_item = n_item

    @abc.abstractmethod
    def v(self, x):
        """
        Valuation for all bidders.
        :param x: vectorized allocation matrix  [(n_bid*n_item),]
        :return: valuations  [n_bid,]
        """
        pass

    @abc.abstractmethod
    def grad_v(self, x):
        """
        Gradient of v for all bidders.
        :return: gradient of v where each row corresponds to a bidder  [n_bid x (n_bid*n_item)]
        """
        pass

    @abc.abstractmethod
    def constraints(self):
        """
        :return: list of scipy.LinearConstraint imposed or assumed by the model.
        """
        pass


class LinearFehrSchmidt(PreferenceModel):
    def __init__(self, a_mat, alpha, beta, ordinal_u):
        """
        For a fixed order over the utilities of bidders (u_i s), FS valuation will be a linear function of utilities.

        :param a_mat: u_i = a_mat[i] * x_i  [n_bid x n_item]
        :param alpha: FS model param  [n_bid,]
        :param beta: FS model param  [n_bid,]
        :param ordinal_u: ordinal u_i of bidders  [n_bid,]
        """
        super().__init__(a_mat.shape[0], a_mat.shape[1])

        self.a_mat = a_mat
        self.alpha = alpha
        self.beta = beta
        self.ordinal_u = ordinal_u

        self.a_tilde_mat = self.a_tilde()

    def a_tilde(self):
        """
        v = a_tilde * x
        :return: a_tilde  [n_bid x (n_bid*n_item)]
        """
        n_bid, n_item = self.a_mat.shape

        a_tilde_mat = np.zeros((n_bid, n_bid*n_item))

        for i_bid in range(n_bid):
            alpha_i = self.alpha[i_bid]
            beta_i = self.beta[i_bid]

            mask_l_i = self.ordinal_u < self.ordinal_u[i_bid]
            mask_g_i = self.ordinal_u > self.ordinal_u[i_bid]

            a_tilde_i_mat = self.a_mat.copy()
            a_tilde_i_mat[mask_l_i] *= beta_i
            a_tilde_i_mat[mask_g_i] *= -alpha_i
            a_tilde_i_mat[i_bid] *= 1 - beta_i*np.sum(mask_l_i) + alpha_i*np.sum(mask_g_i)

            a_tilde_mat[i_bid] = a_tilde_i_mat.reshape((-1,), order='C')

        return a_tilde_mat

    def v(self, x):
        return self.a_tilde_mat.dot(x.reshape((-1, 1)))[:, 0]

    def grad_v(self, _x):
        return self.a_tilde_mat

    def constraints(self):
        n_bid, n_item = self.n_bid, self.n_item

        if n_bid == 1:
            return []

        # Order preserving allocation
        bid_ascend = np.argsort(self.ordinal_u)

        c_ord_pres_mat = np.zeros((n_bid - 1, n_bid*n_item))
        for idx in range(n_bid - 1):
            bid_s = bid_ascend[idx]
            bid_l = bid_ascend[idx + 1]

            c_i = np.zeros((n_bid, n_item))
            c_i[bid_l] = self.a_mat[bid_l]
            c_i[bid_s] = -self.a_mat[bid_s]

            c_ord_pres_mat[idx] = c_i.reshape((-1,), order='C')

        lin_const_ord_pres = LinearConstraint(
            A=c_ord_pres_mat,
            lb=np.zeros((n_bid - 1,)),
            ub=np.inf * np.ones((n_bid - 1,))
        )

        return [lin_const_ord_pres]

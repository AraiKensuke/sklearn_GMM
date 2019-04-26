import numpy as _N
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as _plt

#  generate data  - 2-component Gaussian mixture
N1 = 220
N2 = 380

X              = _N.empty((N1+N2, 1))
X[0:N1, 0]     = 0.1*_N.random.randn(N1)
X[N1:N1+N2, 0] = 5 + 0.1*_N.random.randn(N2)   #  2nd cluster (moveable)


#  max # of components for finite approx of Dirichlet process
n_components = 4
                                     
#  example - setting dof_prior and cov. prior to reasonable value
#  still produces very wide

#  [dof_prior, cov_prior, mean_prec_prior]
prm_sets = [[1., 20, 0.001],
            [100, 1e-9, 0.001]]

fig = _plt.figure(figsize=(7, 3.5))
i_subpl = 0

random_state = 10
BNS=140

#  bins for histogram
xbns   = _N.linspace(-1, 6, BNS+1)
xms    = 0.5*(xbns[0:-1] + xbns[1:])
dx     = _N.diff(xbns)[0]

for prm in prm_sets:
    i_subpl += 1

    occ_cnts, bnsx = _N.histogram(X[:, 0], bins=xbns)
    
    bgm = BayesianGaussianMixture(\
        n_components=n_components,\
        ################  priors
        weight_concentration_prior_type="dirichlet_process",\
        weight_concentration_prior=0.9,\
        degrees_of_freedom_prior=prm[0],\
        covariance_prior=_N.array([prm[1]]),\
        mean_precision_prior=prm[2],\
        ################  priors                              
        reg_covar=0, init_params='random',\
        max_iter=1500,\
        random_state=random_state, covariance_type="diag")

    bgm.fit(X)

    pcs  = bgm.means_.shape[0]

    mns_r  = bgm.means_.T.reshape((1, pcs))
    isd2s_r= bgm.precisions_.T.reshape((1, pcs))
    sd2s_r = bgm.covariances_.T.reshape((1, pcs))
    xms_r  = xms.reshape((BNS, 1))

    A      = (bgm.weights_ / _N.sqrt(2*_N.pi*sd2s_r)) * dx
    occ_x = _N.sum(A*_N.exp(-0.5*(xms_r - mns_r)*(xms_r - mns_r)*isd2s_r), axis=1)
    fig.add_subplot(1, 2, i_subpl)
    _plt.ylim(0, 0.15)
    _plt.plot(xms, occ_cnts/(X.shape[0]))
    _plt.plot(xms, occ_x)
    _plt.title("[%(dof).1f,  %(cov).1e,  %(prc).1f]" % {"dof" : prm[0], "cov" : prm[1], "prc" : prm[2]})

_plt.suptitle("[dof prior,   cov prior,   mn prec prior]")
_plt.savefig("ex3.png")

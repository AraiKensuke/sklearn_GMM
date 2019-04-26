import numpy as _N
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import matplotlib.pyplot as _plt
import time as _tm


t0  = _tm.time()
dat = _N.loadtxt("pos1d.dat")

###  data already filtered to be only during movement
X   = dat.reshape((dat.shape[0], 1))
###  min, max of data for building histogram
max_x = _N.max(dat)
min_x = _N.min(dat)
amp_x = max_x - min_x

###  build histogram of data
BNS=200
xbns   = _N.linspace(min_x, max_x, BNS+1)
xms    = 0.5*(xbns[0:-1] + xbns[1:])
xms_r  = xms.reshape((BNS, 1))
dx     = _N.diff(xbns)[0]

occ_cnts, bnsx = _N.histogram(dat, bins=xbns)

###  count number of peaks in histogram for approximate # of GMM components
docc   = _N.diff(occ_cnts)
peaks  = _N.where((docc[0:-1] > 0) & (docc[1:] < 0))[0]
n_components = len(peaks)+5   #  a few more components than peaks

random_state = int(1000*_N.random.rand())

t1  = _tm.time()
#######
##  Variational Bayes method
#######
bagm = BayesianGaussianMixture\
    (weight_concentration_prior_type="dirichlet_process",\
     weight_concentration_prior=0.05,\
     degrees_of_freedom_prior=0.01,\
     covariance_prior=_N.array([0.01]),\
     tol=1e-2,\
     n_components=n_components, reg_covar=0, init_params='kmeans',\
     max_iter=1500, mean_precision_prior=.001,\
     random_state=random_state, covariance_type="diag")
bagm.fit(X)
t2  = _tm.time()

#######
##  EM method
#######
###  initialize component means  (EM only)
mn_init = _N.empty((n_components, 1))
mn_init[0:len(peaks), 0] = bnsx[peaks]
t3  = _tm.time()
ICs = 5   # number initial conditions   run EM several times, pick best lklhd

###  parmaeter set for each initial condition
em_mns  = _N.empty((ICs, n_components, 1))
em_sd2s = _N.empty((ICs, n_components, 1))
em_wgts = _N.empty((ICs, n_components))
lklhd_ubs= _N.empty(ICs)

###  run EM several times
for ic in range(ICs):
    random_state = int(1000*_N.random.rand())
    emgm = GaussianMixture\
        (n_components=n_components, init_params='kmeans',\
         means_init=mn_init,\
         max_iter=1500,\
         random_state=random_state, covariance_type="diag")
    emgm.fit(X)
    em_mns[ic]  = emgm.means_
    em_sd2s[ic] = emgm.covariances_
    em_wgts[ic] = emgm.weights_
    lklhd_ubs[ic] = emgm.lower_bound_
    
t4  = _tm.time()

print("%(ba).3f  %(em).3f (%(ICs)d repeats)" % {"ba" : (t2-t1), "em" : (t4-t3), "ICs" : ICs})

pcs  = bagm.means_.shape[0]

###
#  calculate estimated pdf using parameters found via VB
###
mns_r  = bagm.means_.T.reshape((1, pcs))
isd2s_r= bagm.precisions_.T.reshape((1, pcs))
sd2s_r = bagm.covariances_.T.reshape((1, pcs))

A_bagm      = (bagm.weights_ / _N.sqrt(2*_N.pi*sd2s_r)) * dx
occ_x_bagm = _N.sum(A_bagm*_N.exp(-0.5*(xms_r - mns_r)*(xms_r - mns_r)*isd2s_r), axis=1)

###
#  calculate estimated pdf using parameters found via EM
###
bestIC = _N.where(lklhd_ubs == _N.max(lklhd_ubs))[0][0]
mns_r  = em_mns[bestIC].T.reshape((1, pcs))
isd2s_r= (1./em_sd2s[bestIC]).T.reshape((1, pcs))
sd2s_r = em_sd2s[bestIC].T.reshape((1, pcs))

A_emgm      = (em_wgts[bestIC] / _N.sqrt(2*_N.pi*sd2s_r)) * dx
occ_x_emgm = _N.sum(A_emgm*_N.exp(-0.5*(xms_r - mns_r)*(xms_r - mns_r)*isd2s_r), axis=1)


####  PLOT
fig = _plt.figure(figsize=(7, 5))
######
_plt.subplot2grid((2, 3), (0, 0))
_plt.plot(xms, occ_cnts/(X.shape[0]), color="black", lw=1.5)
_plt.plot(xms, occ_x_bagm, color="red")
_plt.title("VB")
######
_plt.subplot2grid((2, 3), (1, 0))
_plt.plot(xms, occ_cnts/(X.shape[0]), color="black", lw=1.5)
_plt.plot(xms, occ_x_emgm, color="blue")
ba_w_srtd = _N.sort(bagm.weights_)[::-1]
em_w_srtd = _N.sort(em_wgts[bestIC])[::-1]
_plt.title("EM")
######
_plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
_plt.plot(ba_w_srtd, color="red", label="VB")
_plt.plot(em_w_srtd, color="blue", label="EM")
_plt.xlabel("sorted component #", fontsize=14)
_plt.ylabel("weight", fontsize=14)
_plt.title("Compare EM & VB weights")
fig.subplots_adjust(left=0.15, bottom=0.15, wspace=0.5, hspace=0.5)
_plt.savefig("cmp_VB_EM.png")

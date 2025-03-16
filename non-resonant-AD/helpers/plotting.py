from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np
def plot_all_variables(sig_list, bkg_list, xlabels, labels=["sig", "bkg"], name="sig_vs_bkg", title="", outdir="./", *args, **kwargs):
    csig = 'brown'
    cbkg = 'royalblue'
    
    N = len(sig_list)
    
    if N==len(xlabels):
        fig, ax1 = plt.subplots(1, N, figsize=(6*N,5))
        ax1[0].set_ylabel("Events (A.U.)")
        for i in range(N):
            xmin = np.min(np.hstack([bkg_list[i], sig_list[i]]))
            xmax = np.max(np.hstack([bkg_list[i], sig_list[i]]))
            bins = np.linspace(xmin, xmax, 50)
            ax1[i].hist(sig_list[i], bins = bins, density = True, histtype='step', ls= "-", color=csig, label=labels[0])
            ax1[i].hist(bkg_list[i], bins = bins, density = True, histtype='stepfilled', ls= "-", color=cbkg, alpha=0.5, label=labels[1])
            ax1[i].set_xlabel(xlabels[i])
            ax1[i].set_yticks([])
            ax1[i].legend(loc='upper right', fontsize = 9)

        plt.title(title)
        plt.show()
    else:
        print("Wrong input lists!")
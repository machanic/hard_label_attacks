from collections import OrderedDict

from scipy.special import gammaln
from scipy.special import gamma
from scipy.special import gammaln
import math
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rcParams, rc
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rc('pdf', fonttype=42)

linestyle_dict = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ("dashdot","dashdot"),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

def sign_opt_expectation_gamma(q, d):
    expectation_gamma =  math.sqrt(q) * np.exp(gammaln(d/2.0) - gammaln((d+1)/2.0)) * (1/ math.sqrt(math.pi))
    expectation_gamma_square = 1/float(d) * (2/math.pi * (q - 1) + 1)
    return expectation_gamma, expectation_gamma_square

def prior_sign_opt_expectation_gamma(alpha,q,d):
    expectation_gamma = 1/math.sqrt(q) * (abs(alpha) + (q-1) * math.sqrt(1-alpha ** 2) * np.exp(gammaln((d - 1) / 2) - (gammaln(d / 2))) * (1/ math.sqrt(math.pi)))
    # expectation_gamma_square = 1/float(q) * (alpha**2 + (q-1)/float(d-1) *(2/math.pi*(q-2)+1)*(1-alpha**2)
    #                                          + 2*abs(alpha)*(q-1)*math.sqrt(1-alpha**2)*gamma((d-1)/2)/(gamma(d/2)*math.sqrt(math.pi)))
    expectation_gamma_square = 1 / float(q) * (
                alpha ** 2 + (q - 1) / float(d - 1) * (2 / math.pi * (q - 2) + 1) * (1 - alpha ** 2)
                + 2 * abs(alpha) * (q - 1) * math.sqrt(1 - alpha ** 2) *
                np.exp(gammaln((d - 1) / 2) - (gammaln(d / 2))) / math.sqrt(math.pi))
    return expectation_gamma, expectation_gamma_square

def prior_opt_expectation_gamma_square(alpha, q, d):
    expectation_gamma_square = alpha ** 2 + 1/(d-1) * (2/math.pi * (q-2)+1)*(1-alpha**2)
    return expectation_gamma_square


if __name__ == '__main__':
    d = 3072
    # d = 224*224*3
    alpha_2 = 0.2
    alpha_5 = 0.1

    q_list =np.arange(10,1020,30)
    # q_list = np.arange(10, 1001, 10)
    prior_sign_opt_gamma_square_list_2 = []
    prior_opt_gamma_square_list_2 = []
    prior_sign_opt_gamma_square_list_5 = []
    prior_opt_gamma_square_list_5 = []
    sign_opt_gamma_square_list = []
    for q in q_list:
        sign_opt_gamma_square = sign_opt_expectation_gamma(q, d)[1]
        sign_opt_gamma_square_list.append(sign_opt_gamma_square)
        prior_sign_opt_gamma_square_2 = prior_sign_opt_expectation_gamma(alpha_2,q,d)[1]
        prior_opt_gamma_square_2 = prior_opt_expectation_gamma_square(alpha_2,q,d)
        prior_sign_opt_gamma_square_list_2.append(prior_sign_opt_gamma_square_2)
        prior_opt_gamma_square_list_2.append(prior_opt_gamma_square_2)

        prior_sign_opt_gamma_square_5 = prior_sign_opt_expectation_gamma(alpha_5, q, d)[1]
        prior_opt_gamma_square_5 = prior_opt_expectation_gamma_square(alpha_5, q, d)
        prior_sign_opt_gamma_square_list_5.append(prior_sign_opt_gamma_square_5)
        prior_opt_gamma_square_list_5.append(prior_opt_gamma_square_5)
    x = q_list
    y_sign_opt = np.array(sign_opt_gamma_square_list)
    y_prior_sign_opt_2 = np.array(prior_sign_opt_gamma_square_list_2)
    y_prior_opt_2 = np.array(prior_opt_gamma_square_list_2)
    y_prior_sign_opt_5 = np.array(prior_sign_opt_gamma_square_list_5)
    y_prior_opt_5 = np.array(prior_opt_gamma_square_list_5)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    colors = ['b', 'r', 'c', 'm', 'y', 'k', 'orange', "pink", "brown", "slategrey", "cornflowerblue",
              "greenyellow", "darkgoldenrod", "r", "slategrey", "navy", "darkseagreen", "xkcd:blueberry", "grey",
              "indigo",
              "olivedrab"]
    markers = ['o', '+', '*', 's', "P", "p", "X", "h", "D", "H", "^", "<", "d", ".",  "x", "v", "1", "2", "3", "4"]
    linestyles = [ "dashed", "densely dotted","solid", "dashdotdotted", "densely dashed", "densely dashdotdotted"]

    xtick = np.array([0,100,200,300,400,500,600,700,800,900,1000])
    max_y = max(np.max(y_prior_opt_2).item(), np.max(y_prior_opt_5).item())
    plt.plot(x, y_sign_opt, label="Sign-OPT", color=colors[0],
                 linestyle=linestyle_dict[linestyles[0]], linewidth=1.5,
                 marker=markers[0], markersize=3)
    plt.plot(x, y_prior_sign_opt_2, label=r"Prior-Sign-OPT$_{\alpha=0.2}$", color=colors[1],
             linestyle=linestyle_dict[linestyles[1]], linewidth=1.5,
             marker=markers[1], markersize=3)
    plt.plot(x, y_prior_opt_2, label=r"Prior-OPT$_{\alpha=0.2}$", color=colors[2],
             linestyle=linestyle_dict[linestyles[2]], linewidth=1.5,
             marker=markers[2], markersize=3)

    plt.plot(x, y_prior_sign_opt_5, label=r"Prior-Sign-OPT$_{\alpha=0.1}$", color=colors[3],
             linestyle=linestyle_dict[linestyles[3]], linewidth=1.5,
             marker=markers[3], markersize=3)
    plt.plot(x, y_prior_opt_5, label=r"Prior-OPT$_{\alpha=0.1}$", color=colors[4],
             linestyle=linestyle_dict[linestyles[4]], linewidth=1.5,
             marker=markers[4], markersize=3)

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.ylim(0, max_y)
    plt.gcf().subplots_adjust(bottom=0.15)
    print("max y is {}".format(max_y))
    plt.xticks(xtick, xtick, fontsize=25)
    plt.yticks(np.linspace(0, max_y,11),fontsize=25)
    plt.xlabel(r"$q$", fontsize=25)
    plt.ylabel(r"$\mathbb{E}[\gamma^2]$", fontsize=25)
    plt.legend(loc='upper left',  prop={'size': 23},handlelength=3, framealpha=0.7, fancybox=True, frameon=True)
    plt.tight_layout()
    plt.savefig("D:\\黑盒攻击论文\\hard-label attacks\\Prior-OPT\\ICLR 2025\\figures\\ablation_study\\E_gamma_square\\query_vs_gamma_d={}.pdf".format(d), dpi=200)
    plt.close()




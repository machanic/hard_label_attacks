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
from matplotlib.ticker import FormatStrFormatter


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
    expectation_gamma_lower_bound = np.sqrt(alpha ** 2 + (q-1)*(1-alpha**2)/math.pi * (np.exp(gammaln((d - 1) / 2) - (gammaln(d / 2))))**2)
    expectation_gamma_upper_bound = np.sqrt(alpha ** 2 + 1/(d-1) * (2/math.pi * (q-2)+1)*(1-alpha**2))
    return expectation_gamma_lower_bound, expectation_gamma_upper_bound


if __name__ == '__main__':
    # q = 200
    d = 3072
    # d = 224 * 224 * 3
    q = 50
    if d == 32*32*3:
        alpha_detail_range = np.linspace(0,0.2,10)
    elif d == 224 * 224 * 3:
        alpha_detail_range = np.linspace(0, 0.02, 10)

    sign_opt_gamma_square = sign_opt_expectation_gamma(q,d)[0]
    alpha_list =np.linspace(0,1,51)
    prior_sign_opt_gamma_square_list = []
    prior_opt_gamma_upper_bound_list = []
    prior_opt_gamma_lower_bound_list = []
    for alpha in alpha_list:
        prior_sign_opt_gamma_square = prior_sign_opt_expectation_gamma(alpha,q,d)[0]
        prior_opt_gamma_lower_bound = prior_opt_expectation_gamma_square(alpha, q, d)[0]
        prior_opt_gamma_upper_bound = prior_opt_expectation_gamma_square(alpha,q,d)[1]
        prior_sign_opt_gamma_square_list.append(prior_sign_opt_gamma_square)
        prior_opt_gamma_upper_bound_list.append(prior_opt_gamma_upper_bound)
        prior_opt_gamma_lower_bound_list.append(prior_opt_gamma_lower_bound)
    x = alpha_list
    y_sign_opt = np.array([sign_opt_gamma_square for _ in alpha_list])
    y_prior_sign_opt = np.array(prior_sign_opt_gamma_square_list)
    y_prior_opt_upper_bound = np.array(prior_opt_gamma_upper_bound_list)
    y_prior_opt_lower_bound = np.array(prior_opt_gamma_lower_bound_list)
    print(y_prior_opt_upper_bound-y_prior_opt_lower_bound)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    colors = ['b', 'r', 'c', 'm', 'y', 'k', 'orange', "pink", "brown", "slategrey", "cornflowerblue",
              "greenyellow", "darkgoldenrod", "r", "slategrey", "navy", "darkseagreen", "xkcd:blueberry", "grey",
              "indigo",
              "olivedrab"]
    markers = ['o', '.', '*', 's', "P", "p", "X", "h", "D", "H", "^", "<", "d", ".", "+", "x", "v", "1", "2", "3", "4"]
    linestyles = [ "dashed", "densely dotted","solid", "dashdotdotted", "densely dashed", "densely dashdotdotted"]

    xtick = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    max_y = max(np.max(y_prior_sign_opt).item(), np.max(y_prior_opt_upper_bound).item())
    min_y = 999
    plt.plot(x, y_sign_opt, label="Sign-OPT", color=colors[0],
                 linestyle=linestyle_dict[linestyles[0]], linewidth=1,
                 marker=markers[0], markersize=3)
    plt.plot(x, y_prior_sign_opt, label="Prior-Sign-OPT", color=colors[1],
             linestyle=linestyle_dict[linestyles[1]], linewidth=1,
             marker=markers[1], markersize=3)
    # plt.plot(x, y_prior_opt, label="Prior-OPT", color=colors[2],
    #          linestyle=linestyle_dict[linestyles[2]], linewidth=1,
    #          marker=markers[2], markersize=3)
    plt.fill_between(x, y_prior_opt_lower_bound, y_prior_opt_upper_bound, color=colors[2], label="Prior-OPT")

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.ylim(0, max_y+0.01)
    plt.gcf().subplots_adjust(bottom=0.15)
    print("max y is {}".format(max_y))
    plt.xticks(xtick, xtick, fontsize=20)
    plt.yticks(np.linspace(0, max_y,11), fontsize=20)
    plt.xlabel(r"$\alpha$", fontsize=25)
    plt.ylabel(r"$\mathbb{E}[\gamma]$", fontsize=25)
    plt.legend(loc='upper left', prop={'size': 20}, framealpha=0.5, fancybox=True, frameon=True)
    ax = plt.gca()

    # axins = inset_axes(ax, width="50%", height="50%", loc='upper left',
    #                    bbox_to_anchor=(0.07, 0, 1, 1),
    #                    bbox_transform=ax.transAxes)
    # # axins = ax.inset_axes((0.06, 0.5, 0.6, 0.5))
    # # if d == 200:
    # axins.plot(x, y_sign_opt, label="Sign-OPT", color=colors[0],
    #          linestyle=linestyle_dict[linestyles[0]], linewidth=1,
    #          marker=markers[0], markersize=3)
    # axins.plot(x, y_prior_sign_opt, label="Prior-Sign-OPT", color=colors[1],
    #          linestyle=linestyle_dict[linestyles[1]], linewidth=1,
    #          marker=markers[1], markersize=3)
    # axins.plot(x, y_prior_opt, label="Prior-OPT", color=colors[2],
    #          linestyle=linestyle_dict[linestyles[2]], linewidth=1,
    #          marker=markers[2], markersize=3)
    # axins.set_xticks([0,0.05,0.10,0.15,0.20], [0,0.05,0.10,0.15,0.20])
    # axins.set_yticks([0,0.02,0.04,0.06],[0,0.02,0.04,0.06])
    # axins.set_xlim(0, 0.2)
    # axins.set_ylim(0, 0.1)
    # elif d==224*224*3:
    #     alpha_list = np.linspace(0, 0.02, 21)
    #     prior_sign_opt_gamma_square_list = []
    #     prior_opt_gamma_square_list = []
    #     for alpha in alpha_list:
    #         prior_sign_opt_gamma_square = prior_sign_opt_expectation_gamma(alpha, q, d)[1]
    #         prior_opt_gamma_square = prior_opt_expectation_gamma_square(alpha, q, d)
    #         prior_sign_opt_gamma_square_list.append(prior_sign_opt_gamma_square)
    #         prior_opt_gamma_square_list.append(prior_opt_gamma_square)
    #     x = alpha_list
    #     y_sign_opt = np.array([sign_opt_gamma_square for _ in alpha_list])
    #     y_prior_sign_opt = np.array(prior_sign_opt_gamma_square_list)
    #     y_prior_opt = np.array(prior_opt_gamma_square_list)
    #
    #
    #     axins.plot(x, y_sign_opt, label="Sign-OPT", color=colors[0],
    #                linestyle=linestyle_dict[linestyles[0]], linewidth=1,
    #                marker=markers[0], markersize=3)
    #     axins.plot(x, y_prior_sign_opt, label="Prior-Sign-OPT", color=colors[1],
    #                linestyle=linestyle_dict[linestyles[1]], linewidth=1,
    #                marker=markers[1], markersize=3)
    #     axins.plot(x, y_prior_opt, label="Prior-OPT", color=colors[2],
    #                linestyle=linestyle_dict[linestyles[2]], linewidth=1,
    #                marker=markers[2], markersize=3)
    #     axins.set_xticks([0, 0.005, 0.010, 0.015,0.020],[0, 0.005, 0.010, 0.015,0.020])
    #     axins.set_yticks([0,0.00020,0.00040,0.00060,0.00080],[0,0.00020,0.00040,0.0006,0.00080])
    #     axins.set_xlim(0, 0.02)
    #     axins.set_ylim(0, 0.0008)
    # axins.set_facecolor("lightgrey")
    # axins.set_xticklabels(axins.get_xticks(),color='steelblue')
    # axins.set_yticklabels(axins.get_yticks(), color='steelblue')
    # axins.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axins.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)
    # plt.show()
    plt.savefig("./alpha_vs_gamma_q={}_d={}.pdf".format(q,d), dpi=200)
    plt.close()




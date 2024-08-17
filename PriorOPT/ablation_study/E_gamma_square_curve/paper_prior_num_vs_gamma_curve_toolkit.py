import random
from collections import OrderedDict

import numpy
from matplotlib.ticker import StrMethodFormatter
from scipy.special import gammaln
import math
import numpy as np
import matplotlib.pyplot as plt

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

def prior_sign_opt_expectation_gamma(alpha_list,q,d):
    s = len(alpha_list)
    expectation_gamma = 1/math.sqrt(q) * (
            sum(abs(alpha) for alpha in alpha_list)
            + (q-s) * math.sqrt(1- sum(alpha ** 2 for alpha in alpha_list)) *
            np.exp(gammaln((d - s) / 2) - (gammaln((d-s+1) / 2))) / math.sqrt(math.pi))

    expectation_gamma_square = 1 / float(q) * (
                sum(abs(alpha) for alpha in alpha_list) ** 2 + (q - s) / float(d - s) * (2 / math.pi * (q - s-1) + 1) * (1 - sum(alpha ** 2 for alpha in alpha_list))
                + 2 * sum(abs(alpha) for alpha in alpha_list)  * (q - s) * math.sqrt(1 - sum(alpha ** 2 for alpha in alpha_list))  *
                np.exp(gammaln((d - s) / 2) - (gammaln((d-s+1) / 2))) / math.sqrt(math.pi))
    return expectation_gamma, expectation_gamma_square

def prior_opt_expectation_gamma_square(alpha_list, q, d):
    s = len(alpha_list)
    expectation_gamma_square = sum(alpha**2 for alpha in alpha_list) + 1.0/(d-s)*(2/math.pi*(q-s-1)+1)*(1-sum(alpha**2 for alpha in alpha_list))
    return expectation_gamma_square


def generate_alpha_list(s):
    lst = []
    sum_of_squares = 0

    while sum_of_squares <= 1 and len(lst) < s:
        remaining_sum = 1 - sum_of_squares
        num = random.uniform(0, math.sqrt(remaining_sum))
        if num < 0.2 and num>1e-5:
            lst.append(num)
            sum_of_squares += num**2

    return lst


if __name__ == '__main__':
    q = 50
    d = 3072
    # d = 224 * 224 * 3
    if d == 32*32*3:
        max_alpha = 0.2
    elif d == 224 * 224 * 3:
        max_alpha = 0.03
    sign_opt_gamma_square = sign_opt_expectation_gamma(q,d)[1]
    prior_sign_opt_gamma_square_list = []
    prior_opt_gamma_square_list = []
    sorted_prior_sign_opt_gamma_square_list = []
    sorted_prior_opt_gamma_square_list = []
    alpha_list = generate_alpha_list(20)
    reverse_sorted_alpha_list = sorted(alpha_list,reverse=True)
    prior_number = 20
    for s in range(1,prior_number+1):
        # uniform_alpha_list = [max_alpha/2.0 for _ in range(s)]
        random_alpha_list = alpha_list[:s]
        reversed_alpha_list = reverse_sorted_alpha_list[:s]
        prior_sign_opt_gamma_square_list.append(prior_sign_opt_expectation_gamma(random_alpha_list,q,d)[1])
        prior_opt_gamma_square_list.append(prior_opt_expectation_gamma_square(random_alpha_list,q,d))
        sorted_prior_sign_opt_gamma_square_list.append(prior_sign_opt_expectation_gamma(reversed_alpha_list,q,d)[1])
        sorted_prior_opt_gamma_square_list.append(prior_opt_expectation_gamma_square(reversed_alpha_list,q,d))

    x = [s for s in range(1,prior_number+1)]
    y_sign_opt = np.array([sign_opt_gamma_square for _ in x])
    y_prior_sign_opt = np.array(prior_sign_opt_gamma_square_list)
    y_prior_opt = np.array(prior_opt_gamma_square_list)
    y_prior_sign_opt_sorted = np.array(sorted_prior_sign_opt_gamma_square_list)
    y_prior_opt_sorted = np.array(sorted_prior_opt_gamma_square_list)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    colors = ['b', 'r', 'c', 'm', 'y', 'k', 'orange', "pink", "brown", "slategrey", "cornflowerblue",
              "greenyellow", "darkgoldenrod", "r", "slategrey", "navy", "darkseagreen", "xkcd:blueberry", "grey",
              "indigo",
              "olivedrab"]
    markers = ['o', '.', '*', 's', "P", "p", "X", "h", "D", "H", "^", "<", "d", ".", "+", "x", "v", "1", "2", "3", "4"]
    linestyles = [ "dashed", "densely dotted","solid", "dashdotdotted", "densely dashed", "densely dashdotdotted"]

    xtick = np.arange(1,prior_number+1,2)
    max_y = max(np.max(y_prior_sign_opt).item(), np.max(y_prior_opt).item())
    min_y = 999
    plt.plot(x, y_sign_opt, label="Sign-OPT", color=colors[0],
                 linestyle=linestyle_dict[linestyles[0]], linewidth=1.5,
                 marker=markers[0], markersize=5)
    plt.plot(x, y_prior_sign_opt, label=r"Prior-Sign-OPT$_{\mathregular{Shuffle}\{\alpha_1,\alpha_2,\cdots,\alpha_s\}}$", color=colors[1],
             linestyle=linestyle_dict[linestyles[1]], linewidth=1.5,
             marker=markers[1], markersize=5)
    plt.plot(x, y_prior_opt, label=r"Prior-OPT$_{\mathregular{Shuffle}\{\alpha_1,\alpha_2,\cdots,\alpha_s\}}$", color=colors[2],
             linestyle=linestyle_dict[linestyles[2]], linewidth=1.5,
             marker=markers[2], markersize=5)
    plt.plot(x, y_prior_sign_opt_sorted, label=r"Prior-Sign-OPT$_{\alpha_1\geq\alpha_2\geq\cdots\geq\alpha_s}$", color=colors[3],
             linestyle=linestyle_dict[linestyles[3]], linewidth=1.5,
             marker=markers[3], markersize=5)
    plt.plot(x, y_prior_opt_sorted, label=r"Prior-OPT$_{\alpha_1\geq\alpha_2\geq\cdots\geq\alpha_s}$", color=colors[4],
             linestyle=linestyle_dict[linestyles[4]], linewidth=1.5,
             marker=markers[4], markersize=5)

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.ylim(0, max_y+0.01)
    plt.gcf().subplots_adjust(bottom=0.15)
    print("max y is {}".format(max_y))
    plt.xticks(xtick, xtick, fontsize=20)
    plt.yticks(np.linspace(0, max_y,11), fontsize=20)
    plt.xlabel(r"Number of Priors", fontsize=25)
    plt.ylabel(r"$\mathbb{E}[\gamma^2]$", fontsize=25)
    plt.legend(loc='upper left', prop={'size': 20}, framealpha=0.5, fancybox=True, frameon=True)
    ax = plt.gca()
    #
    # # axins = inset_axes(ax, width="50%", height="40%", loc='upper left',
    # #                    bbox_to_anchor=(0.05, 0.05, 1, 1),
    # #                    bbox_transform=ax.transAxes)
    # axins = ax.inset_axes((0.06, 0.5, 0.6, 0.5))
    # if d == 32 * 32 * 3:
    #     axins.plot(x, y_sign_opt, label="Sign-OPT", color=colors[0],
    #              linestyle=linestyle_dict[linestyles[0]], linewidth=1,
    #              marker=markers[0], markersize=3)
    #     axins.plot(x, y_prior_sign_opt, label="Prior-Sign-OPT", color=colors[1],
    #              linestyle=linestyle_dict[linestyles[1]], linewidth=1,
    #              marker=markers[1], markersize=3)
    #     axins.plot(x, y_prior_opt, label="Prior-OPT", color=colors[2],
    #              linestyle=linestyle_dict[linestyles[2]], linewidth=1,
    #              marker=markers[2], markersize=3)
    #     axins.set_xticks([0,0.05,0.10,0.15,0.20], [0,0.05,0.10,0.15,0.20], fontsize=13)
    #     axins.set_yticks([0,0.02,0.04,0.06,0.08],[0,0.02,0.04,0.06,0.08],fontsize=13)
    #     axins.set_xlim(0, 0.2)
    #     axins.set_ylim(0, 0.08)
    # elif d==224*224*3:
    #     alpha_list = np.linspace(0, 0.03, 21)
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
    #     axins.set_xticks([0, 0.005, 0.010, 0.015,0.020,0.025, 0.030],[0, 0.005, 0.010, 0.015,0.020,0.025, 0.030], fontsize=13)
    #     axins.set_yticks([0,0.0005,0.001, 0.0015,0.002],[0,0.0005,0.001, 0.0015,0.002], fontsize=13)
    #     axins.set_xlim(0, 0.03)
    #     axins.set_ylim(0, 0.002)
    # axins.set_facecolor("lightgrey")
    # axins.set_xticklabels(axins.get_xticks(),color='steelblue')
    # axins.set_yticklabels(axins.get_yticks(), color='steelblue')
    # mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)
    plt.savefig("D:\\黑盒攻击论文\\hard-label attacks\\Prior-OPT\\NeurIPS 2024\\figures\\ablation_study\\E_gamma_square\\prior_num_vs_gamma_q={}_d={}.pdf".format(q,d), dpi=200)
    plt.close()




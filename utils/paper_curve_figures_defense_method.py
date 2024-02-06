import os
import random
import sys
sys.path.append(os.getcwd())
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import make_interp_spline
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
from config import MODELS_TEST_STANDARD

from matplotlib import rcParams, rc
# rcParams['xtick.direction'] = 'out'
# rcParams['ytick.direction'] = 'out'
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rc('pdf', fonttype=42)

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

def longest_common_subsequence(x: str, y: str):
    """
    Finds the longest common subsequence between two strings. Also returns the
    The subsequence found

    Parameters
    ----------

    x: str, one of the strings
    y: str, the other string

    Returns
    -------
    L[m][n]: int, the length of the longest subsequence. Also equal to len(seq)
    Seq: str, the subsequence found

    """
    # find the length of strings

    assert x is not None
    assert y is not None

    m = len(x)
    n = len(y)

    # declaring the array for storing the dp values
    l = [[0] * (n + 1) for _ in range(m + 1)]  # noqa: E741

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = 1 if x[i - 1] == y[j - 1] else 0

            l[i][j] = max(l[i - 1][j], l[i][j - 1], l[i - 1][j - 1] + match)

    seq = ""
    i, j = m, n
    while i > 0 and j > 0:
        match = 1 if x[i - 1] == y[j - 1] else 0

        if l[i][j] == l[i - 1][j - 1] + match:
            if match == 1:
                seq = x[i - 1] + seq
            i -= 1
            j -= 1
        elif l[i][j] == l[i - 1][j]:
            i -= 1
        else:
            j -= 1

    return l[m][n], seq


def read_json_data(json_path):
    # data_key can be query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query
    print("begin read {}".format(json_path))
    with open(json_path, "r") as file_obj:
        data_txt = file_obj.read()
        data_json = json.loads(data_txt)
        distortion_dict = data_json["distortion"]
        surrogate_archs = []
        if "surrogate_arch" in data_json["args"] and data_json["args"]["surrogate_arch"] is not None:
            surrogate_archs.append(data_json["args"]["surrogate_arch"])
        elif "surrogate_archs" in data_json["args"] and data_json["args"]["surrogate_archs"] is not None:
            surrogate_archs.extend(data_json["args"]["surrogate_archs"])
    return distortion_dict,  surrogate_archs

surrogate_arch_name_to_paper = {"inceptionresnetv2":"IncResV2", "xception":"Xception", "resnet50":"ResNet50","convit_base":"ConViT",
                                "jx_vit":"ViT", "resnet-110":"ResNet110", "senet154":"SENet154", "densenet-bc-100-12":"DenseNetBC100"}
def read_all_data(dataset_path_dict, arch, query_budgets, stats="mean_distortion"):
    # dataset_path_dict {("CIFAR-10","l2","untargeted"): "/.../"， }
    data_info = {}
    for (dataset, norm, targeted, method), dir_path in dataset_path_dict.items():
        for file_path in os.listdir(dir_path):
            if longest_common_subsequence(arch, file_path)[1] == arch and file_path.endswith(".json"):
                file_path = dir_path + "/" + file_path
                distortion_dict, surrogate_archs = read_json_data(file_path)
                x = []
                y = []
                for query_budget in query_budgets:
                    if not targeted and method == "BA" and query_budget<1000:
                        print("Skip boundary attack at {} query budgets".format(query_budget))
                        continue
                    distortion_list = []
                    for image_id, query_distortion_dict in distortion_dict.items():
                        query_distortion_dict = {int(float(query)): float(dist) for query, dist in query_distortion_dict.items()}
                        queries = np.array(list(query_distortion_dict.keys()))
                        queries = np.sort(queries)
                        find_index = np.searchsorted(queries, query_budget, side='right') - 1
                        if query_budget < queries[find_index]:
                            print(
                                "query budget is {}, find query is {}, min query is {}, len query_distortion is {}".format(
                                    query_budget, queries[find_index], np.min(queries).item(),
                                    len(query_distortion_dict)))
                            continue
                        distortion_list.append(query_distortion_dict[queries[find_index]])
                    distortion_list = np.array(distortion_list)
                    distortion_list = distortion_list[~np.isnan(distortion_list)]  # 去掉nan的值
                    mean_distortion = np.mean(distortion_list)
                    median_distortion = np.median(distortion_list)
                    x.append(query_budget)
                    if stats == "mean_distortion":
                        y.append(mean_distortion)
                    elif stats == "median_distortion":
                        y.append(median_distortion)
                x = np.array(x)
                y = np.array(y)
                surrogate_archs_new = [surrogate_arch_name_to_paper[surrogate_arch] for surrogate_arch in
                                       surrogate_archs]
                data_info[(dataset, norm, targeted, method, "&".join(surrogate_archs_new))] = (x,y)
    return data_info




method_name_to_paper = {"tangent_attack":"TA",
                        "ellipsoid_tangent_attack":"G-TA", "GeoDA":"GeoDA",
                        "HSJA":"HSJA", "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT",
                         "Evolutionary":"Evolutionary", "SurFree":"SurFree",
                        "TriangleAttack":"Triangle Attack", "PriorSignOPT":"Prior-Sign-OPT",
                        "PriorOPT":"Prior-OPT", "RayS":"RayS"
                        #"QEBA":"QEBA", "CGBA_H":"CGBA-H"
                        }
# method_name_to_paper = {
#                         "PriorOPT":"Prior-OPT","RayS":"RayS"
#                         #"QEBA":"QEBA", "CGBA_H":"CGBA-H"
#                         }
def from_method_to_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack" or method == "ellipsoid_tangent_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "GeoDA" or method == "RayS" or method == "SurFree" or method == "Evolutionary" or method == "TriangleAttack" or method == "PriorOPT" or method == "PriorSignOPT":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "biased_boundary_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "boundary_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "RayS":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SignOPT":
        if targeted:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        if targeted:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    return path

def get_all_exists_folder(dataset, methods, norm, targeted):
    root_dir = "F:/logs/hard_label_attack_complete/"
    dataset_path_dict = {}  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): "/.../"， }
    for method in methods:
        file_name = from_method_to_dir_path(dataset, method, norm, targeted)
        file_path = root_dir + file_name
        if os.path.exists(file_path):
            dataset_path_dict[(dataset, norm, targeted, method_name_to_paper[method])] = file_path
        else:
            print("{} does not exist!!!".format(file_path))
    return dataset_path_dict

method_linestyle_mark_dict = {}
def draw_query_distortion_figure(dataset, norm, targeted, arch, fig_type, dump_file_path, xlabel, ylabel):

    # fig_type can be [query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query]
    methods = list(method_name_to_paper.keys())
    if targeted:
        methods.remove("TriangleAttack")
        methods.remove("RayS")
    if norm == "l2":
        methods.remove("RayS")
    elif norm == "linf":
        methods.remove("TriangleAttack")
        methods.remove("ellipsoid_tangent_attack")
        methods.remove("tangent_attack")
        methods.remove("SurFree")
        methods.remove("Evolutionary")

    dataset_path_dict= get_all_exists_folder(dataset, methods, norm, targeted)
    max_query = 10000
    if dataset=="ImageNet":
        max_query = 20000
    query_budgets = np.arange(1000, max_query+1, 1000)
    data_info = read_all_data(dataset_path_dict, arch, query_budgets, fig_type)  # fig_type can be mean_distortion or median_distortion
    plt.style.use('bmh')
    # from matplotlib import rcParams, rc
    # rcParams['xtick.direction'] = 'out'
    # rcParams['ytick.direction'] = 'out'
    # rcParams['pdf.fonttype'] = 42
    # rcParams['ps.fonttype'] = 42
    # rc('pdf', fonttype=42)

    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'orange', "pink", "brown", "slategrey", "r", "greenyellow",
               "cornflowerblue",  "steelblue", "navy", "darkseagreen","peru","xkcd:blueberry"]


    markers = ['o', '>', '*', 's', "P", "p", "X", "h", "D", "H", "^", "<", ".", "+", "x","v"]
    linestyles = ["solid", "dashed", "densely dotted", "dashdotdotted", "densely dashed", "densely dashdotdotted"]


    xtick = np.array([ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    if max_query == 20000:
        xtick = np.array([ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000])
    max_y = 0
    min_y= 999
    for idx, ((dataset, norm, targeted, method, surrogate_archs), (x,y)) in enumerate(data_info.items()):
        x = np.asarray(x)
        y = np.asarray(y)
        if np.max(y) > max_y:
            max_y = np.max(y)
        if np.min(y) < min_y:
            min_y = np.min(y)
        if surrogate_archs:
            method = method + "$_{\mathrm{"+ surrogate_archs + "}}$"
        if method not in method_linestyle_mark_dict:
            linestyle = linestyle_dict[linestyles[idx % len(linestyles)]]
            mark = markers[idx]
            color = colors[idx]
            while (mark, color) in [(tuple_value[1],tuple_value[2]) for tuple_value in method_linestyle_mark_dict.values()]:
                color = colors[random.randint(0, len(colors)-1)]
                mark = markers[random.randint(0, len(markers)-1)]
            method_linestyle_mark_dict[method] = (linestyle, mark, color)
        plt.plot(x, y, label=method, color=method_linestyle_mark_dict[method][2],
                         linestyle=method_linestyle_mark_dict[method][0], linewidth=1.5,
                         marker=method_linestyle_mark_dict[method][1], markersize=6)

    if norm == "linf":
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    else:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    if dataset == "ImageNet":
        plt.xlim(0, max_query + 1000)
    else:
        plt.xlim(0, max_query)
    plt.ylim(0, max_y+0.1)
    plt.gcf().subplots_adjust(bottom=0.15)
    print("max y is {}".format(max_y))

    if dataset == "ImageNet":
        x_ticks = xtick[0::2]
        x_ticks = x_ticks.tolist()
        x_ticks_label = ["0"] + ["{}K".format(x_tick // 1000) for x_tick in x_ticks[1:]]
        plt.xticks(x_ticks, x_ticks_label, fontsize=20)
    else:
        x_ticks_label = ["0"] + ["{}K".format(x_tick // 1000) for x_tick in xtick[1:]]
        plt.xticks(xtick, x_ticks_label, fontsize=20)
    if dataset=="ImageNet":
        if norm == "l2":
            yticks = np.arange(0, max_y + 1, 5)
        else:
            yticks = np.linspace(0, max_y+0.1,11)
    else:
        if norm == "l2":
            yticks = np.arange(0, max_y+1)
        else:
            yticks = np.linspace(0, max_y+0.1, 11)
    plt.yticks(yticks, fontsize=20)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)

    if "jpeg" in dump_file_path:
        plt.legend(loc='upper right', prop={'size': 15},
                   handlelength=4,framealpha=0.5,fancybox=True,frameon=True)
    elif "ImageNet" == dataset:
        plt.legend(loc='lower left', prop={'size': 15},
                   handlelength=4,framealpha=0.5,fancybox=True,frameon=True)
    elif ("com_defend" in dump_file_path or "feature_distillation" in dump_file_path) and "CIFAR-10" == dataset and targeted:
        plt.legend(loc='upper right', prop={'size': 15},handlelength=4,framealpha=0.5,fancybox=True,frameon=True)
    else:
        plt.legend(loc='upper right', prop={'size': 15},handlelength=4,framealpha=0.5,fancybox=True,frameon=True)
    plt.savefig(dump_file_path, dpi=200)
    plt.close()
    print("save to {}".format(dump_file_path))

def parse_args():
    parser = argparse.ArgumentParser(description='Drawing Figures of Attacking Normal Models')
    parser.add_argument("--fig_type", type=str, choices=["mean_distortion",
                                                         "median_distortion"])
    parser.add_argument("--dataset", type=str, help="the dataset to train")
    parser.add_argument("--norm", type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    dump_folder = "D:/黑盒攻击论文/hard-label attacks/Prior-OPT/icml2024/figures/query_vs_distortion/"
    os.makedirs(dump_folder, exist_ok=True)
    for dataset in ["CIFAR-10"]:
        for norm in ["linf"]:
            args.norm = norm
            for targeted in [False]:
                if "CIFAR" in dataset:
                    archs = ['resnet-50_TRADES', "resnet-50_jpeg", "resnet-50_feature_scatter",
                             "resnet-50_feature_distillation", "resnet-50_com_defend", "resnet-50_adv_train"]
                else:
                    archs= ["resnet50_adv_train_on_ImageNet_linf_4_div_255", "resnet50_adv_train_on_ImageNet_l2_3",
                            "resnet50_adv_train_on_ImageNet_linf_8_div_255"]
                for model in archs:
                    file_path = dump_folder + "{dataset}_{model}_{norm}_{target_str}_attack.pdf".format(dataset=dataset,
                                  model=model, norm=args.norm, target_str="untargeted" if not targeted else "targeted")
                    x_label = "Number of Queries"
                    if args.fig_type == "mean_distortion" and args.norm == "l2":
                        y_label = "Mean $\ell_2$ Distortion"
                    elif args.fig_type == "median_distortion" and args.norm == "l2":
                        y_label = "Median $\ell_2$ Distortion"
                    elif args.fig_type == "mean_distortion" and args.norm == "linf":
                        y_label = "Mean $\ell_\infty$ Distortion"
                    elif args.fig_type == "median_distortion" and args.norm == "linf":
                        y_label = "Median $\ell_\infty$ Distortion"
                    draw_query_distortion_figure(dataset, args.norm, targeted, model, args.fig_type, file_path,x_label,y_label)

        # elif args.fig_type == "query_hist":
        #     target_str = "/untargeted" if not args.targeted else "targeted"
        #     os.makedirs(dump_folder, exist_ok=True)
        #     for dataset in ["CIFAR-10","CIFAR-100", "TinyImageNet"]:
        #         if "CIFAR" in dataset:
        #             archs = ['pyramidnet272', "gdas", "WRN-28-10-drop", "WRN-40-10-drop"]
        #         else:
        #             archs = ["densenet121", "resnext32_4", "resnext64_4"]
        #         for norm in ["l2","linf"]:
        #             for model in archs:
        #                 draw_histogram_fig(dataset, norm, args.targeted, model, dump_folder + target_str)

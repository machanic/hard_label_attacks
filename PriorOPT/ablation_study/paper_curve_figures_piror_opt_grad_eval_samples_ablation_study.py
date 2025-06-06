import os
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

from matplotlib.ticker import StrMethodFormatter
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


def read_json_data(json_path):
    # data_key can be query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query
    print("begin read {}".format(json_path))
    with open(json_path, "r") as file_obj:
        data_txt = file_obj.read()
        data_json = json.loads(data_txt)
        distortion_dict = data_json["distortion"]
    return distortion_dict

def read_all_data(dataset_path_dict, arch, query_budgets, grad_est_samples, stats="mean_distortion"):
    data_info = {}
    for (dataset, norm, targeted, method), dir_path in dataset_path_dict.items():
        for grad_est_sample in grad_est_samples:
            for file_path in os.listdir(dir_path):
                if file_path.startswith(arch) and file_path.endswith(".json") and "with_{}_grad_samples".format(grad_est_sample) in file_path:
                    file_path = dir_path + "/" + file_path
                    distortion_dict = read_json_data(file_path)
                    x = []
                    y = []
                    for query_budget in query_budgets:
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
                    data_info[(dataset, norm, targeted, method, grad_est_sample)] = (x,y)
                    break
    return data_info




method_name_to_paper = {"tangent_attack":"TA",
                        "ellipsoid_tangent_attack":"G-TA", "GeoDA":"GeoDA",
                        "HSJA":"HSJA",  "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT",
                         "Evolutionary":"Evolutionary", "SurFree":"SurFree",
                        "TriangleAttack":"TriA", "PriorSignOPT":"Prior-Sign-OPT",
                        "PriorOPT":"Prior-OPT", "RayS":"RayS"
                        #"QEBA":"QEBA", "CGBA_H":"CGBA-H"
                        }

def from_method_to_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJARandom":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "GeoDA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "biased_boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "RayS":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    return path

def get_all_exists_folder(dataset, method, norm, targeted):
    root_dir = "G:/logs/hard_label_attacks/"
    dataset_path_dict = {}  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): "/.../"， }
    file_name = from_method_to_dir_path(dataset, method, norm, targeted)
    file_path = root_dir + file_name +"/ablation_study"
    if os.path.exists(file_path):
        dataset_path_dict[(dataset, norm, targeted, method_name_to_paper[method])] = file_path
    else:
        print("{} does not exist!!!".format(file_path))
    return dataset_path_dict



def draw_query_distortion_figure(dataset, norm, targeted, arch, fig_type, dump_file_path, xlabel, ylabel):

    # fig_type can be [query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query]
    method = "PriorSignOPT"
    dataset_path_dict= get_all_exists_folder(dataset, method, norm, targeted)
    max_query = 10000
    if dataset=="ImageNet" and targeted:
        max_query = 20000
    query_budgets = np.arange(1000, max_query+1, 1000)
    data_info = read_all_data(dataset_path_dict, arch, query_budgets,np.arange(20,201,20).tolist(), fig_type)  # fig_type can be mean_distortion or median_distortion
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'orange', "pink", "brown", "slategrey", "cornflowerblue", "greenyellow",
              "maroon", "steelblue", "r", "slategrey", "navy","olive","darkred","orchid"]
    markers = ['o', '>', '*', 's', "P", "p", "X", "h", "D", "H", "^", "<","v",".",  "$\mathrm{o}$",  "+", "x","1","2","3"]
    linestyles = ["solid", "dashed", "densely dotted", "dashdotdotted", "densely dashed", "densely dashdotdotted","loosely dashed","dashdot"]

    xtick = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    if max_query == 20000:
        xtick = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000])
    max_y = 0
    min_y= 999
    for idx, ((dataset, norm, targeted, method, num_eval_grad), (x,y)) in enumerate(data_info.items()):
        color = colors[idx]
        x = np.asarray(x)
        y = np.asarray(y)
        if np.max(y) > max_y:
            max_y = np.max(y)
        if np.min(y) < min_y:
            min_y = np.min(y)
        line, = plt.plot(x, y, label="$q={}$".format(num_eval_grad), color=color,
                         linestyle=linestyle_dict[linestyles[idx%len(linestyles)]], linewidth=1.0,
                         marker=markers[idx], markersize=6)
    if dataset!="ImageNet":
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    else:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    if dataset == "ImageNet" and targeted:
        plt.xlim(0, max_query+1000)
    else:
        plt.xlim(0, max_query)

    plt.ylim(0, max_y+0.1)
    plt.gcf().subplots_adjust(bottom=0.15)
    print("max y is {}".format(max_y))
    # xtick = [0, 5000, 10000]
    if dataset == "ImageNet" and targeted:
        x_ticks = xtick[0::2]
        x_ticks = x_ticks.tolist()
        x_ticks_label = ["0"] + ["{}K".format(x_tick // 1000) for x_tick in x_ticks[1:]]
        plt.xticks(x_ticks, x_ticks_label, fontsize=25)
    else:
        x_ticks_label = ["0"] + ["{}K".format(x_tick // 1000) for x_tick in xtick[1:]]
        plt.xticks(xtick, x_ticks_label, fontsize=25)
    if dataset == "ImageNet":
        yticks = np.arange(0, max_y + 1, 5)
    else:
        yticks = np.arange(0, max_y + 0.1, 0.5)
    plt.yticks(yticks, fontsize=25)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(loc='upper right', prop={'size': 25},handlelength=4,framealpha=0.4,fancybox=True,frameon=True,ncol=2)
    plt.tight_layout()
    plt.savefig(dump_file_path, dpi=200)
    plt.close()
    print("save to {}".format(dump_file_path))



def parse_args():
    parser = argparse.ArgumentParser(description='Drawing Figures of Attacking Normal Models')
    parser.add_argument("--fig_type", type=str, choices=["mean_distortion",
                                                         "median_distortion"])
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to train")
    parser.add_argument("--norm", type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dump_folder = "D:/黑盒攻击论文/hard-label attacks/Prior-OPT/ICLR 2025/figures/ablation_study/"
    os.makedirs(dump_folder, exist_ok=True)

    if "CIFAR" in args.dataset:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        archs = ["swin_base_patch4_window7_224"]

    for model in archs:
        file_path  = dump_folder + "{dataset}_{model}_{norm}_{target_str}_attack.pdf".format(dataset=args.dataset,
                      model=model, norm=args.norm, target_str="untargeted" if not args.targeted else "targeted")
        x_label = "Number of Queries"
        if args.fig_type == "mean_distortion":
            y_label = "Mean $\ell_2$ Distortion"
        elif args.fig_type == "median_distortion":
            y_label = "Median $\ell_2$ Distortion"

        draw_query_distortion_figure(args.dataset, args.norm, args.targeted, model, args.fig_type, file_path,x_label,y_label)
        # draw_random_HSJA_TangentAttack_query_distortion_figure(args.dataset, args.norm, args.targeted,
        #                                                         model, args.fig_type, file_path, x_label,y_label)
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

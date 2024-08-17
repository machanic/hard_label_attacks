from collections import defaultdict

import bisect
import numpy as np
import json
import os

def new_round(_float, _len):
    """
    Parameters
    ----------
    _float: float
    _len: int, 指定四舍五入需要保留的小数点后几位数为_len

    Returns
    -------
    type ==> float, 返回四舍五入后的值
    """
    if isinstance(_float, float):
        if str(_float)[::-1].find('.') <= _len:
            return (_float)
        if str(_float)[-1] == '5':
            return (round(float(str(_float)[:-1] + '6'), _len))
        else:
            return (round(_float, _len))
    else:
        return (round(_float, _len))


method_name_to_paper = {"tangent_attack":"TA",  "HSJA":"HSJA",
                        "ellipsoid_tangent_attack":"G-TA",
                        "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT",
                        "biased_boundary_attack":"BBA",
                        "QEBA":"QEBA","Evolutionary":"Evolutionary",
                        "SQBA":"SQBA","GeoDA":"GeoDA","TriangleAttack":"Triangle Attack",
                        "SurFree":"SurFree",
                        "PriorSignOPT_1prior": "Prior-Sign-OPT-1prior",
                        "PriorSignOPT_2prior": "Prior-Sign-OPT-2prior",
                        "PriorSignOPT_PGD_init_theta":"Prior-Sign-OPT_PGD_init_theta",
                        "PriorOPT_1prior": "Prior-OPT-1prior",
                        "PriorOPT_2prior": "Prior-OPT-2prior",
                        "PriorOPT_PGD_init_theta":"Prior-OPT_PGD_init_theta",
                        }


# method_name_to_paper = {
#                         "SignOPT":"Sign-OPT",
#                         "PriorSignOPT": "Prior-Sign-OPT",
#                         "PriorOPT": "Prior-OPT"
#                         }


def from_method_to_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "ellipsoid_tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "GeoDA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "RayS":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                               norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "biased_boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "AHA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "Evolutionary":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "TriangleAttack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "CGBA_H":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SQBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SurFree":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT_1prior":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT_1prior":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT_PGD_init_theta":
        path = "PriorSignOPT-{dataset}-{norm}-{target_str}_with_PGD_init_theta".format( dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT_2prior":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT_2prior":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT_PGD_init_theta":
        path = "PriorOPT-{dataset}-{norm}-{target_str}_with_PGD_init_theta".format( dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "QEBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    return path


def read_json_and_extract(json_path):
    with open(json_path, "r") as file_obj:
        json_content = json.load(file_obj)
        distortion = json_content["distortion"]
        return distortion

def get_file_name_list(dataset, method_name_to_paper, norm, targeted):
    folder_path_dict = {}
    for method, paper_method_name in method_name_to_paper.items():
        file_path = "H:/logs/hard_label_attack_complete/" + from_method_to_dir_path(dataset, method, norm, targeted)
        folder_path_dict[paper_method_name] = file_path
    return folder_path_dict

def bin_search(arr, target):
    if target not in arr:
        return None
    arr.sort()
    return arr[arr.index(target)-1], arr.index(target)-1

def get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets, want_key):
    mean_and_median_distortions = defaultdict(lambda : "-")
    for query_budget in query_budgets:
        distortion_list = []
        for idx, (image_index, query_distortion) in enumerate(distortion_dict.items()):
            if idx==0:
                assert int(image_index) == 0
            query_distortion = {int(float(query)):float(dist) for query,dist in query_distortion.items()}
            queries = np.array(list(query_distortion.keys()))
            queries = np.sort(queries)
            find_index = np.searchsorted(queries, query_budget, side='right') - 1
            if query_budget < queries[find_index]:
                print("query budget is {}, find query is {}, min query is {}, len query_distortion is {}".format(query_budget, queries[find_index], np.min(queries).item(), len(query_distortion)))
                continue
            distortion_list.append(query_distortion[queries[find_index]])
        distortion_list = np.array(distortion_list)
        distortion_list = distortion_list[~np.isnan(distortion_list)]  # 去掉nan的值
        mean_distortion = np.mean(distortion_list)
        median_distortion = np.median(distortion_list)
        if want_key == "mean_distortion":
            mean_and_median_distortions[query_budget] = "{:.3f}".format(new_round(mean_distortion.item(),3))
        elif want_key =="median_distortion":
            mean_and_median_distortions[query_budget] = "{:.3f}".format(new_round(median_distortion.item(),3))
    return mean_and_median_distortions


def fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, query_budgets,
    wanted_one_prior_list, wanted_two_prior_list, want_key="mean_distortion"):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result = defaultdict(lambda : defaultdict(lambda : "-"))
    for method, folder in folder_list.items():
        if not os.path.exists(folder):
            continue
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if file_name.startswith(arch) and file_name.endswith(".json"):
                if method in ["RayS","GeoDA"] and targeted:
                    print("{} does not exist!".format(file_path))
                    result[method] = defaultdict(lambda : "-")
                    continue
                if not os.path.exists(file_path):
                    distortion_dict = {}
                else:
                    if method.endswith("1prior"):
                        json_surrogate_arch = []
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            if json_content["args"]["surrogate_arch"]:
                                json_surrogate_arch.append(json_content["args"]["surrogate_arch"])
                            elif json_content["args"]["surrogate_archs"]:
                                json_surrogate_arch.extend(json_content["args"]["surrogate_archs"])
                            distortion_dict = json_content["distortion"]
                        if len(json_surrogate_arch) != 1 or json_surrogate_arch[0] not in wanted_one_prior_list:
                            distortion_dict.clear()
                            continue
                        print("Read  : " + file_path)
                    elif method.endswith("2prior"):
                        json_surrogate_arch = []
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            if json_content["args"]["surrogate_archs"]:
                                json_surrogate_arch.extend(json_content["args"]["surrogate_archs"])
                            distortion_dict = json_content["distortion"]
                        two_prior_found = False
                        if len(json_surrogate_arch) != 2:
                            distortion_dict.clear()
                            continue
                        for two_prior in wanted_two_prior_list:
                            if len(set(two_prior) & set(json_surrogate_arch)) == 2:
                                two_prior_found = True
                        if not two_prior_found:
                            distortion_dict.clear()
                            continue
                        print("Read  : " + file_path)
                    else:
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            distortion_dict = json_content["distortion"]
                            print("Read  : " + file_path)

                mean_and_median_distortions = get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets,want_key)
                result[method] = mean_and_median_distortions
    return result





def fetch_json_content_for_multiple_priors(dataset, norm, targeted, arch, query_budgets, want_key="mean_distortion"):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result = defaultdict(lambda : defaultdict(lambda : "-"))
    for method, folder in folder_list.items():
        if not os.path.exists(folder):
            continue
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if file_name.startswith(arch) and file_name.endswith(".json"):
                if method in ["RayS","GeoDA"] and targeted:
                    print("{} does not exist!".format(file_path))
                    result[method] = defaultdict(lambda : "-")
                    continue
                if not os.path.exists(file_path):
                    distortion_dict = {}
                else:
                    if method.startswith("Prior"):
                        json_surrogate_arch = []
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            if json_content["args"]["surrogate_archs"]:
                                json_surrogate_arch.extend(json_content["args"]["surrogate_archs"])
                            else:
                                json_surrogate_arch.append(json_content["args"]["surrogate_arch"])
                            distortion_dict = json_content["distortion"]
                        new_method = method+"-{}prior".format(len(json_surrogate_arch))
                        print("{}  : ".format(new_method) + file_path + " Archs:{}".format( ",".join(json_surrogate_arch)))
                    else:
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            distortion_dict = json_content["distortion"]
                            print("Read  : " + file_path)
                        new_method = method

                mean_and_median_distortions = get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets,want_key)
                result[new_method] = mean_and_median_distortions
    return result

def draw_tables_for_multi_priors(result, surrogate_1, surrogate_2, surrogate_3, surrogate_4, surrogate_5):
    print("""
                & Sign-OPT \\cite{{cheng2019sign}}& no prior & {SignOPT_1000_victim_model_1} & {SignOPT_2000_victim_model_1} & {SignOPT_5000_victim_model_1} & {SignOPT_8000_victim_model_1} & {SignOPT_10000_victim_model_1}  & {SignOPT_1000_victim_model_2} & {SignOPT_2000_victim_model_2} & {SignOPT_5000_victim_model_2} & {SignOPT_8000_victim_model_2} & {SignOPT_10000_victim_model_2} \\\\
                & Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}}} & 1 prior & {PriorSignOPT_1prior_1000_victim_model_1} & {PriorSignOPT_1prior_2000_victim_model_1} & {PriorSignOPT_1prior_5000_victim_model_1} & {PriorSignOPT_1prior_8000_victim_model_1} & {PriorSignOPT_1prior_10000_victim_model_1}  & {PriorSignOPT_1prior_1000_victim_model_2} & {PriorSignOPT_1prior_2000_victim_model_2} & {PriorSignOPT_1prior_5000_victim_model_2} & {PriorSignOPT_1prior_8000_victim_model_2} & {PriorSignOPT_1prior_10000_victim_model_2} \\
                & Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}} & 2 priors  & {PriorSignOPT_2prior_1000_victim_model_1} & {PriorSignOPT_2prior_2000_victim_model_1} & {PriorSignOPT_2prior_5000_victim_model_1} & {PriorSignOPT_2prior_8000_victim_model_1} & {PriorSignOPT_2prior_10000_victim_model_1}  & {PriorSignOPT_2prior_1000_victim_model_2} & {PriorSignOPT_2prior_2000_victim_model_2} & {PriorSignOPT_2prior_5000_victim_model_2} & {PriorSignOPT_2prior_8000_victim_model_2} & {PriorSignOPT_2prior_10000_victim_model_2} \\
                & Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}\&{surrogate_3}}} & 3 priors & {PriorSignOPT_3prior_1000_victim_model_1} & {PriorSignOPT_3prior_2000_victim_model_1} & {PriorSignOPT_3prior_5000_victim_model_1} & {PriorSignOPT_3prior_8000_victim_model_1} & {PriorSignOPT_3prior_10000_victim_model_1}  & {PriorSignOPT_3prior_1000_victim_model_2} & {PriorSignOPT_3prior_2000_victim_model_2} & {PriorSignOPT_3prior_5000_victim_model_2} & {PriorSignOPT_3prior_8000_victim_model_2} & {PriorSignOPT_3prior_10000_victim_model_2} \\
                & Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}\&{surrogate_3}\&{surrogate_4}}} & 4 priors & {PriorSignOPT_4prior_1000_victim_model_1} & {PriorSignOPT_4prior_2000_victim_model_1} & {PriorSignOPT_4prior_5000_victim_model_1} & {PriorSignOPT_4prior_8000_victim_model_1} & {PriorSignOPT_4prior_10000_victim_model_1}  & {PriorSignOPT_4prior_1000_victim_model_2} & {PriorSignOPT_4prior_2000_victim_model_2} & {PriorSignOPT_4prior_5000_victim_model_2} & {PriorSignOPT_4prior_8000_victim_model_2} & {PriorSignOPT_4prior_10000_victim_model_2} \\
                & Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}\&{surrogate_3}\&{surrogate_4}\&{surrogate_5}}} & 5 priors & {PriorSignOPT_5prior_1000_victim_model_1} & {PriorSignOPT_5prior_2000_victim_model_1} & {PriorSignOPT_5prior_5000_victim_model_1} & {PriorSignOPT_5prior_8000_victim_model_1} & {PriorSignOPT_5prior_10000_victim_model_1}  & {PriorSignOPT_5prior_1000_victim_model_2} & {PriorSignOPT_5prior_2000_victim_model_2} & {PriorSignOPT_5prior_5000_victim_model_2} & {PriorSignOPT_5prior_8000_victim_model_2} & {PriorSignOPT_5prior_10000_victim_model_2} \\
                & Prior-OPT\\textsubscript{{\\tiny {surrogate_1}}} & 1 prior & {PriorOPT_1prior_1000_victim_model_1} & {PriorOPT_1prior_2000_victim_model_1} & {PriorOPT_1prior_5000_victim_model_1} & {PriorOPT_1prior_8000_victim_model_1} & {PriorOPT_1prior_10000_victim_model_1}  & {PriorOPT_1prior_1000_victim_model_2} & {PriorOPT_1prior_2000_victim_model_2} & {PriorOPT_1prior_5000_victim_model_2} & {PriorOPT_1prior_8000_victim_model_2} & {PriorOPT_1prior_10000_victim_model_2}\\\\
                & Prior-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}} & 2 priors & {PriorOPT_2prior_1000_victim_model_1} & {PriorOPT_2prior_2000_victim_model_1} & {PriorOPT_2prior_5000_victim_model_1} & {PriorOPT_2prior_8000_victim_model_1} & {PriorOPT_2prior_10000_victim_model_1}  & {PriorOPT_2prior_1000_victim_model_2} & {PriorOPT_2prior_2000_victim_model_2} & {PriorOPT_2prior_5000_victim_model_2} & {PriorOPT_2prior_8000_victim_model_2} & {PriorOPT_2prior_10000_victim_model_2} \\\\
                & Prior-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}\&{surrogate_3}}} & 3 priors & {PriorOPT_3prior_1000_victim_model_1} & {PriorOPT_3prior_2000_victim_model_1} & {PriorOPT_3prior_5000_victim_model_1} & {PriorOPT_3prior_8000_victim_model_1} & {PriorOPT_3prior_10000_victim_model_1}  & {PriorOPT_3prior_1000_victim_model_2} & {PriorOPT_3prior_2000_victim_model_2} & {PriorOPT_3prior_5000_victim_model_2} & {PriorOPT_3prior_8000_victim_model_2} & {PriorOPT_3prior_10000_victim_model_2} \\\\
                & Prior-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}\&{surrogate_3}\&{surrogate_4}}} & 4 priors & {PriorOPT_4prior_1000_victim_model_1} & {PriorOPT_4prior_2000_victim_model_1} & {PriorOPT_4prior_5000_victim_model_1} & {PriorOPT_4prior_8000_victim_model_1} & {PriorOPT_4prior_10000_victim_model_1}  & {PriorOPT_4prior_1000_victim_model_2} & {PriorOPT_4prior_2000_victim_model_2} & {PriorOPT_4prior_5000_victim_model_2} & {PriorOPT_4prior_8000_victim_model_2} & {PriorOPT_4prior_10000_victim_model_2} \\\\
                & Prior-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}\&{surrogate_3}\&{surrogate_4}\&{surrogate_5}}} & 5 priors & {PriorOPT_5prior_1000_victim_model_1} & {PriorOPT_5prior_2000_victim_model_1} & {PriorOPT_5prior_5000_victim_model_1} & {PriorOPT_5prior_8000_victim_model_1} & {PriorOPT_5prior_10000_victim_model_1}  & {PriorOPT_5prior_1000_victim_model_2} & {PriorOPT_5prior_2000_victim_model_2} & {PriorOPT_5prior_5000_victim_model_2} & {PriorOPT_5prior_8000_victim_model_2} & {PriorOPT_5prior_10000_victim_model_2} \\\\
                """.format(
        surrogate_1=surrogate_1,
        surrogate_2=surrogate_2,
        surrogate_3=surrogate_3,
        surrogate_4=surrogate_4,
        surrogate_5=surrogate_5,

        SignOPT_1000_victim_model_1=result[False]["Sign-OPT"][1000],
        SignOPT_2000_victim_model_1=result[False]["Sign-OPT"][2000],
        SignOPT_5000_victim_model_1=result[False]["Sign-OPT"][5000],
        SignOPT_8000_victim_model_1=result[False]["Sign-OPT"][8000],
        SignOPT_10000_victim_model_1=result[False]["Sign-OPT"][10000],


        PriorSignOPT_1prior_1000_victim_model_1=result[False]["Prior-Sign-OPT-1prior"][1000],
        PriorSignOPT_1prior_2000_victim_model_1=result[False]["Prior-Sign-OPT-1prior"][2000],
        PriorSignOPT_1prior_5000_victim_model_1=result[False]["Prior-Sign-OPT-1prior"][5000],
        PriorSignOPT_1prior_8000_victim_model_1=result[False]["Prior-Sign-OPT-1prior"][8000],
        PriorSignOPT_1prior_10000_victim_model_1=result[False]["Prior-Sign-OPT-1prior"][10000],

        PriorSignOPT_2prior_1000_victim_model_1=result[False]["Prior-Sign-OPT-2prior"][1000],
        PriorSignOPT_2prior_2000_victim_model_1=result[False]["Prior-Sign-OPT-2prior"][2000],
        PriorSignOPT_2prior_5000_victim_model_1=result[False]["Prior-Sign-OPT-2prior"][5000],
        PriorSignOPT_2prior_8000_victim_model_1=result[False]["Prior-Sign-OPT-2prior"][8000],
        PriorSignOPT_2prior_10000_victim_model_1=result[False]["Prior-Sign-OPT-2prior"][10000],

        PriorSignOPT_3prior_1000_victim_model_1=result[False]["Prior-Sign-OPT-3prior"][1000],
        PriorSignOPT_3prior_2000_victim_model_1=result[False]["Prior-Sign-OPT-3prior"][2000],
        PriorSignOPT_3prior_5000_victim_model_1=result[False]["Prior-Sign-OPT-3prior"][5000],
        PriorSignOPT_3prior_8000_victim_model_1=result[False]["Prior-Sign-OPT-3prior"][8000],
        PriorSignOPT_3prior_10000_victim_model_1=result[False]["Prior-Sign-OPT-3prior"][10000],

        PriorSignOPT_4prior_1000_victim_model_1=result[False]["Prior-Sign-OPT-4prior"][1000],
        PriorSignOPT_4prior_2000_victim_model_1=result[False]["Prior-Sign-OPT-4prior"][2000],
        PriorSignOPT_4prior_5000_victim_model_1=result[False]["Prior-Sign-OPT-4prior"][5000],
        PriorSignOPT_4prior_8000_victim_model_1=result[False]["Prior-Sign-OPT-4prior"][8000],
        PriorSignOPT_4prior_10000_victim_model_1=result[False]["Prior-Sign-OPT-4prior"][10000],

        PriorSignOPT_5prior_1000_victim_model_1=result[False]["Prior-Sign-OPT-5prior"][1000],
        PriorSignOPT_5prior_2000_victim_model_1=result[False]["Prior-Sign-OPT-5prior"][2000],
        PriorSignOPT_5prior_5000_victim_model_1=result[False]["Prior-Sign-OPT-5prior"][5000],
        PriorSignOPT_5prior_8000_victim_model_1=result[False]["Prior-Sign-OPT-5prior"][8000],
        PriorSignOPT_5prior_10000_victim_model_1=result[False]["Prior-Sign-OPT-5prior"][10000],

        PriorOPT_1prior_1000_victim_model_1=result[False]["Prior-OPT-1prior"][1000],
        PriorOPT_1prior_2000_victim_model_1=result[False]["Prior-OPT-1prior"][2000],
        PriorOPT_1prior_5000_victim_model_1=result[False]["Prior-OPT-1prior"][5000],
        PriorOPT_1prior_8000_victim_model_1=result[False]["Prior-OPT-1prior"][8000],
        PriorOPT_1prior_10000_victim_model_1=result[False]["Prior-OPT-1prior"][10000],

        PriorOPT_2prior_1000_victim_model_1=result[False]["Prior-OPT-2prior"][1000],
        PriorOPT_2prior_2000_victim_model_1=result[False]["Prior-OPT-2prior"][2000],
        PriorOPT_2prior_5000_victim_model_1=result[False]["Prior-OPT-2prior"][5000],
        PriorOPT_2prior_8000_victim_model_1=result[False]["Prior-OPT-2prior"][8000],
        PriorOPT_2prior_10000_victim_model_1=result[False]["Prior-OPT-2prior"][10000],

        PriorOPT_3prior_1000_victim_model_1=result[False]["Prior-OPT-3prior"][1000],
        PriorOPT_3prior_2000_victim_model_1=result[False]["Prior-OPT-3prior"][2000],
        PriorOPT_3prior_5000_victim_model_1=result[False]["Prior-OPT-3prior"][5000],
        PriorOPT_3prior_8000_victim_model_1=result[False]["Prior-OPT-3prior"][8000],
        PriorOPT_3prior_10000_victim_model_1=result[False]["Prior-OPT-3prior"][10000],

        PriorOPT_4prior_1000_victim_model_1=result[False]["Prior-OPT-4prior"][1000],
        PriorOPT_4prior_2000_victim_model_1=result[False]["Prior-OPT-4prior"][2000],
        PriorOPT_4prior_5000_victim_model_1=result[False]["Prior-OPT-4prior"][5000],
        PriorOPT_4prior_8000_victim_model_1=result[False]["Prior-OPT-4prior"][8000],
        PriorOPT_4prior_10000_victim_model_1=result[False]["Prior-OPT-4prior"][10000],

        PriorOPT_5prior_1000_victim_model_1=result[False]["Prior-OPT-5prior"][1000],
        PriorOPT_5prior_2000_victim_model_1=result[False]["Prior-OPT-5prior"][2000],
        PriorOPT_5prior_5000_victim_model_1=result[False]["Prior-OPT-5prior"][5000],
        PriorOPT_5prior_8000_victim_model_1=result[False]["Prior-OPT-5prior"][8000],
        PriorOPT_5prior_10000_victim_model_1=result[False]["Prior-OPT-5prior"][10000],


        SignOPT_1000_victim_model_2=result[True]["Sign-OPT"][1000],
        SignOPT_2000_victim_model_2=result[True]["Sign-OPT"][2000],
        SignOPT_5000_victim_model_2=result[True]["Sign-OPT"][5000],
        SignOPT_8000_victim_model_2=result[True]["Sign-OPT"][8000],
        SignOPT_10000_victim_model_2=result[True]["Sign-OPT"][10000],


        PriorSignOPT_1prior_1000_victim_model_2=result[True]["Prior-Sign-OPT-1prior"][1000],
        PriorSignOPT_1prior_2000_victim_model_2=result[True]["Prior-Sign-OPT-1prior"][2000],
        PriorSignOPT_1prior_5000_victim_model_2=result[True]["Prior-Sign-OPT-1prior"][5000],
        PriorSignOPT_1prior_8000_victim_model_2=result[True]["Prior-Sign-OPT-1prior"][8000],
        PriorSignOPT_1prior_10000_victim_model_2=result[True]["Prior-Sign-OPT-1prior"][10000],

        PriorSignOPT_2prior_1000_victim_model_2=result[True]["Prior-Sign-OPT-2prior"][1000],
        PriorSignOPT_2prior_2000_victim_model_2=result[True]["Prior-Sign-OPT-2prior"][2000],
        PriorSignOPT_2prior_5000_victim_model_2=result[True]["Prior-Sign-OPT-2prior"][5000],
        PriorSignOPT_2prior_8000_victim_model_2=result[True]["Prior-Sign-OPT-2prior"][8000],
        PriorSignOPT_2prior_10000_victim_model_2=result[True]["Prior-Sign-OPT-2prior"][10000],

        PriorSignOPT_3prior_1000_victim_model_2=result[True]["Prior-Sign-OPT-3prior"][1000],
        PriorSignOPT_3prior_2000_victim_model_2=result[True]["Prior-Sign-OPT-3prior"][2000],
        PriorSignOPT_3prior_5000_victim_model_2=result[True]["Prior-Sign-OPT-3prior"][5000],
        PriorSignOPT_3prior_8000_victim_model_2=result[True]["Prior-Sign-OPT-3prior"][8000],
        PriorSignOPT_3prior_10000_victim_model_2=result[True]["Prior-Sign-OPT-3prior"][10000],

        PriorSignOPT_4prior_1000_victim_model_2=result[True]["Prior-Sign-OPT-4prior"][1000],
        PriorSignOPT_4prior_2000_victim_model_2=result[True]["Prior-Sign-OPT-4prior"][2000],
        PriorSignOPT_4prior_5000_victim_model_2=result[True]["Prior-Sign-OPT-4prior"][5000],
        PriorSignOPT_4prior_8000_victim_model_2=result[True]["Prior-Sign-OPT-4prior"][8000],
        PriorSignOPT_4prior_10000_victim_model_2=result[True]["Prior-Sign-OPT-4prior"][10000],

        PriorSignOPT_5prior_1000_victim_model_2=result[True]["Prior-Sign-OPT-5prior"][1000],
        PriorSignOPT_5prior_2000_victim_model_2=result[True]["Prior-Sign-OPT-5prior"][2000],
        PriorSignOPT_5prior_5000_victim_model_2=result[True]["Prior-Sign-OPT-5prior"][5000],
        PriorSignOPT_5prior_8000_victim_model_2=result[True]["Prior-Sign-OPT-5prior"][8000],
        PriorSignOPT_5prior_10000_victim_model_2=result[True]["Prior-Sign-OPT-5prior"][10000],

        PriorOPT_1prior_1000_victim_model_2=result[True]["Prior-OPT-1prior"][1000],
        PriorOPT_1prior_2000_victim_model_2=result[True]["Prior-OPT-1prior"][2000],
        PriorOPT_1prior_5000_victim_model_2=result[True]["Prior-OPT-1prior"][5000],
        PriorOPT_1prior_8000_victim_model_2=result[True]["Prior-OPT-1prior"][8000],
        PriorOPT_1prior_10000_victim_model_2=result[True]["Prior-OPT-1prior"][10000],

        PriorOPT_2prior_1000_victim_model_2=result[True]["Prior-OPT-2prior"][1000],
        PriorOPT_2prior_2000_victim_model_2=result[True]["Prior-OPT-2prior"][2000],
        PriorOPT_2prior_5000_victim_model_2=result[True]["Prior-OPT-2prior"][5000],
        PriorOPT_2prior_8000_victim_model_2=result[True]["Prior-OPT-2prior"][8000],
        PriorOPT_2prior_10000_victim_model_2=result[True]["Prior-OPT-2prior"][10000],

        PriorOPT_3prior_1000_victim_model_2=result[True]["Prior-OPT-3prior"][1000],
        PriorOPT_3prior_2000_victim_model_2=result[True]["Prior-OPT-3prior"][2000],
        PriorOPT_3prior_5000_victim_model_2=result[True]["Prior-OPT-3prior"][5000],
        PriorOPT_3prior_8000_victim_model_2=result[True]["Prior-OPT-3prior"][8000],
        PriorOPT_3prior_10000_victim_model_2=result[True]["Prior-OPT-3prior"][10000],

        PriorOPT_4prior_1000_victim_model_2=result[True]["Prior-OPT-4prior"][1000],
        PriorOPT_4prior_2000_victim_model_2=result[True]["Prior-OPT-4prior"][2000],
        PriorOPT_4prior_5000_victim_model_2=result[True]["Prior-OPT-4prior"][5000],
        PriorOPT_4prior_8000_victim_model_2=result[True]["Prior-OPT-4prior"][8000],
        PriorOPT_4prior_10000_victim_model_2=result[True]["Prior-OPT-4prior"][10000],

        PriorOPT_5prior_1000_victim_model_2=result[True]["Prior-OPT-5prior"][1000],
        PriorOPT_5prior_2000_victim_model_2=result[True]["Prior-OPT-5prior"][2000],
        PriorOPT_5prior_5000_victim_model_2=result[True]["Prior-OPT-5prior"][5000],
        PriorOPT_5prior_8000_victim_model_2=result[True]["Prior-OPT-5prior"][8000],
        PriorOPT_5prior_10000_victim_model_2=result[True]["Prior-OPT-5prior"][10000],
    )
    )



def draw_tables_for_ImageNet(result, surrogate_1, surrogate_2):
    print("""
                & HSJA \\cite{{chen2019hopskipjumpattack}} & {HSJA_1000_untargeted} & {HSJA_2000_untargeted} & {HSJA_5000_untargeted} & {HSJA_8000_untargeted} & {HSJA_10000_untargeted}  & {HSJA_1000_targeted} & {HSJA_2000_targeted} & {HSJA_5000_targeted} & {HSJA_8000_targeted} & {HSJA_10000_targeted} & {HSJA_15000_targeted} & {HSJA_20000_targeted}\\\\
				& TA \\cite{{ma2021finding}} & {TA_1000_untargeted} & {TA_2000_untargeted} & {TA_5000_untargeted} & {TA_8000_untargeted} & {TA_10000_untargeted}  & {TA_1000_targeted} & {TA_2000_targeted} & {TA_5000_targeted} & {TA_8000_targeted} & {TA_10000_targeted}& {TA_15000_targeted} & {TA_20000_targeted}\\\\
				& G-TA \\cite{{ma2021finding}}  & {GTA_1000_untargeted} & {GTA_2000_untargeted} & {GTA_5000_untargeted} & {GTA_8000_untargeted} & {GTA_10000_untargeted}  & {GTA_1000_targeted} & {GTA_2000_targeted} & {GTA_5000_targeted} & {GTA_8000_targeted} & {GTA_10000_targeted} & {GTA_15000_targeted} & {GTA_20000_targeted}\\\\
				& QEBA \\cite{{li2020qeba}} & {QEBA_1000_untargeted} & {QEBA_2000_untargeted} & {QEBA_5000_untargeted} & {QEBA_8000_untargeted} & {QEBA_10000_untargeted}  & {QEBA_1000_targeted} & {QEBA_2000_targeted} & {QEBA_5000_targeted} & {QEBA_8000_targeted} & {QEBA_10000_targeted} & {QEBA_15000_targeted} & {QEBA_20000_targeted}\\\\
                & Sign-OPT \\cite{{cheng2019sign}} & {SignOPT_1000_untargeted} & {SignOPT_2000_untargeted} & {SignOPT_5000_untargeted} & {SignOPT_8000_untargeted} & {SignOPT_10000_untargeted}  & {SignOPT_1000_targeted} & {SignOPT_2000_targeted} & {SignOPT_5000_targeted} & {SignOPT_8000_targeted} & {SignOPT_10000_targeted} & {SignOPT_15000_targeted} & {SignOPT_20000_targeted}\\\\
				& SVM-OPT \\cite{{cheng2019sign}} & {SVMOPT_1000_untargeted} & {SVMOPT_2000_untargeted} & {SVMOPT_5000_untargeted} & {SVMOPT_8000_untargeted} & {SVMOPT_10000_untargeted}  & {SVMOPT_1000_targeted} & {SVMOPT_2000_targeted} & {SVMOPT_5000_targeted} & {SVMOPT_8000_targeted} & {SVMOPT_10000_targeted}& {SVMOPT_15000_targeted} & {SVMOPT_20000_targeted}\\\\
                & GeoDA \\cite{{rahmati2020geoda}} & {GeoDA_1000_untargeted} & {GeoDA_2000_untargeted} & {GeoDA_5000_untargeted} & {GeoDA_8000_untargeted} & {GeoDA_10000_untargeted}  & {GeoDA_1000_targeted} & {GeoDA_2000_targeted} & {GeoDA_5000_targeted} & {GeoDA_8000_targeted} & {GeoDA_10000_targeted} & {GeoDA_15000_targeted} & {GeoDA_20000_targeted}\\\\
                & Evolutionary \\cite{{dong2019efficient}} & {Evolutionary_1000_untargeted} & {Evolutionary_2000_untargeted} & {Evolutionary_5000_untargeted} & {Evolutionary_8000_untargeted} & {Evolutionary_10000_untargeted}  & {Evolutionary_1000_targeted} & {Evolutionary_2000_targeted} & {Evolutionary_5000_targeted} & {Evolutionary_8000_targeted} & {Evolutionary_10000_targeted}  & {Evolutionary_15000_targeted} & {Evolutionary_20000_targeted}\\\\
                & SurFree \\cite{{maho2021surfree}}  & {SurFree_1000_untargeted} & {SurFree_2000_untargeted} & {SurFree_5000_untargeted} & {SurFree_8000_untargeted} & {SurFree_10000_untargeted}  & {SurFree_1000_targeted} & {SurFree_2000_targeted} & {SurFree_5000_targeted} & {SurFree_8000_targeted} & {SurFree_10000_targeted}  & {SurFree_15000_targeted} & {SurFree_20000_targeted}\\\\
                & Triangle Attack \\cite{{wang2022triangle}} & {TriangleAttack_1000_untargeted} & {TriangleAttack_2000_untargeted} & {TriangleAttack_5000_untargeted} & {TriangleAttack_8000_untargeted} & {TriangleAttack_10000_untargeted}  & {TriangleAttack_1000_targeted} & {TriangleAttack_2000_targeted} & {TriangleAttack_5000_targeted} & {TriangleAttack_8000_targeted} & {TriangleAttack_10000_targeted} & {TriangleAttack_15000_targeted} & {TriangleAttack_20000_targeted}\\\\
                & SQBA \\cite{{Park_2024_sqba}} & {SQBA_1000_untargeted} & {SQBA_2000_untargeted} & {SQBA_5000_untargeted} & {SQBA_8000_untargeted} & {SQBA_10000_untargeted}  & {SQBA_1000_targeted} & {SQBA_2000_targeted} & {SQBA_5000_targeted} & {SQBA_8000_targeted} & {SQBA_10000_targeted}  & {SQBA_15000_targeted} & {SQBA_20000_targeted}\\\\
                & BBA \\cite{{brunner2019guessing}}  & {BBA_1000_untargeted} & {BBA_2000_untargeted} & {BBA_5000_untargeted} & {BBA_8000_untargeted} & {BBA_10000_untargeted}  & {BBA_1000_targeted} & {BBA_2000_targeted} & {BBA_5000_targeted} & {BBA_8000_targeted} & {BBA_10000_targeted}  & {BBA_15000_targeted} & {BBA_20000_targeted}\\\\
                & Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}}} & {PriorSignOPT_1prior_1000_untargeted} & {PriorSignOPT_1prior_2000_untargeted} & {PriorSignOPT_1prior_5000_untargeted} & {PriorSignOPT_1prior_8000_untargeted} & {PriorSignOPT_1prior_10000_untargeted}  & {PriorSignOPT_1prior_1000_targeted} & {PriorSignOPT_1prior_2000_targeted} & {PriorSignOPT_1prior_5000_targeted} & {PriorSignOPT_1prior_8000_targeted} & {PriorSignOPT_1prior_10000_targeted} & {PriorSignOPT_1prior_15000_targeted} & {PriorSignOPT_1prior_20000_targeted} \\\\
                & Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}}   & {PriorSignOPT_2prior_1000_untargeted} & {PriorSignOPT_2prior_2000_untargeted} & {PriorSignOPT_2prior_5000_untargeted} & {PriorSignOPT_2prior_8000_untargeted} & {PriorSignOPT_2prior_10000_untargeted}  & {PriorSignOPT_2prior_1000_targeted} & {PriorSignOPT_2prior_2000_targeted} & {PriorSignOPT_2prior_5000_targeted} & {PriorSignOPT_2prior_8000_targeted} & {PriorSignOPT_2prior_10000_targeted} & {PriorSignOPT_2prior_15000_targeted} & {PriorSignOPT_2prior_20000_targeted} \\\\
                & Prior-Sign-OPT\\textsubscript{{\\tiny $\\theta_0^\\text{{PGD}}$ + {surrogate_1}}}  & {PriorSignOPT_PGD_init_theta_1000_untargeted} & {PriorSignOPT_PGD_init_theta_2000_untargeted} & {PriorSignOPT_PGD_init_theta_5000_untargeted} & {PriorSignOPT_PGD_init_theta_8000_untargeted} & {PriorSignOPT_PGD_init_theta_10000_untargeted}  & {PriorSignOPT_PGD_init_theta_1000_targeted} & {PriorSignOPT_PGD_init_theta_2000_targeted} & {PriorSignOPT_PGD_init_theta_5000_targeted} & {PriorSignOPT_PGD_init_theta_8000_targeted} & {PriorSignOPT_PGD_init_theta_10000_targeted}  & {PriorSignOPT_PGD_init_theta_15000_targeted} & {PriorSignOPT_PGD_init_theta_20000_targeted} \\\\
                & Prior-OPT\\textsubscript{{\\tiny {surrogate_1}}} & {PriorOPT_1prior_1000_untargeted} & {PriorOPT_1prior_2000_untargeted} & {PriorOPT_1prior_5000_untargeted} & {PriorOPT_1prior_8000_untargeted} & {PriorOPT_1prior_10000_untargeted}  & {PriorOPT_1prior_1000_targeted} & {PriorOPT_1prior_2000_targeted} & {PriorOPT_1prior_5000_targeted} & {PriorOPT_1prior_8000_targeted} & {PriorOPT_1prior_10000_targeted} & {PriorOPT_1prior_15000_targeted} & {PriorOPT_1prior_20000_targeted}\\\\
                & Prior-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}} & {PriorOPT_2prior_1000_untargeted} & {PriorOPT_2prior_2000_untargeted} & {PriorOPT_2prior_5000_untargeted} & {PriorOPT_2prior_8000_untargeted} & {PriorOPT_2prior_10000_untargeted}  & {PriorOPT_2prior_1000_targeted} & {PriorOPT_2prior_2000_targeted} & {PriorOPT_2prior_5000_targeted} & {PriorOPT_2prior_8000_targeted} & {PriorOPT_2prior_10000_targeted} & {PriorOPT_2prior_15000_targeted} & {PriorOPT_2prior_20000_targeted} \\\\
                & Prior-OPT\\textsubscript{{\\tiny $\\theta_0^\\text{{PGD}}$ + {surrogate_1}}} & {PriorOPT_PGD_init_theta_1000_untargeted} & {PriorOPT_PGD_init_theta_2000_untargeted} & {PriorOPT_PGD_init_theta_5000_untargeted} & {PriorOPT_PGD_init_theta_8000_untargeted} & {PriorOPT_PGD_init_theta_10000_untargeted}  & {PriorOPT_PGD_init_theta_1000_targeted} & {PriorOPT_PGD_init_theta_2000_targeted} & {PriorOPT_PGD_init_theta_5000_targeted} & {PriorOPT_PGD_init_theta_8000_targeted} & {PriorOPT_PGD_init_theta_10000_targeted} & {PriorOPT_PGD_init_theta_15000_targeted} & {PriorOPT_PGD_init_theta_20000_targeted} \\\\ 
                """.format(
        surrogate_1=surrogate_1,
        surrogate_2=surrogate_2,

        BBA_1000_untargeted=result[False]["BBA"][1000],
        BBA_2000_untargeted=result[False]["BBA"][2000],
        BBA_5000_untargeted=result[False]["BBA"][5000],
        BBA_8000_untargeted=result[False]["BBA"][8000],
        BBA_10000_untargeted=result[False]["BBA"][10000],

        HSJA_1000_untargeted=result[False]["HSJA"][1000],
        HSJA_2000_untargeted=result[False]["HSJA"][2000],
        HSJA_5000_untargeted=result[False]["HSJA"][5000],
        HSJA_8000_untargeted=result[False]["HSJA"][8000],
        HSJA_10000_untargeted=result[False]["HSJA"][10000],

        TA_1000_untargeted=result[False]["TA"][1000],
        TA_2000_untargeted=result[False]["TA"][2000],
        TA_5000_untargeted=result[False]["TA"][5000],
        TA_8000_untargeted=result[False]["TA"][8000],
        TA_10000_untargeted=result[False]["TA"][10000],

        GTA_1000_untargeted=result[False]["G-TA"][1000],
        GTA_2000_untargeted=result[False]["G-TA"][2000],
        GTA_5000_untargeted=result[False]["G-TA"][5000],
        GTA_8000_untargeted=result[False]["G-TA"][8000],
        GTA_10000_untargeted=result[False]["G-TA"][10000],

        QEBA_1000_untargeted=result[False]["QEBA"][1000],
        QEBA_2000_untargeted=result[False]["QEBA"][2000],
        QEBA_5000_untargeted=result[False]["QEBA"][5000],
        QEBA_8000_untargeted=result[False]["QEBA"][8000],
        QEBA_10000_untargeted=result[False]["QEBA"][10000],

        SignOPT_1000_untargeted=result[False]["Sign-OPT"][1000],
        SignOPT_2000_untargeted=result[False]["Sign-OPT"][2000],
        SignOPT_5000_untargeted=result[False]["Sign-OPT"][5000],
        SignOPT_8000_untargeted=result[False]["Sign-OPT"][8000],
        SignOPT_10000_untargeted=result[False]["Sign-OPT"][10000],

        SVMOPT_1000_untargeted=result[False]["SVM-OPT"][1000],
        SVMOPT_2000_untargeted=result[False]["SVM-OPT"][2000],
        SVMOPT_5000_untargeted=result[False]["SVM-OPT"][5000],
        SVMOPT_8000_untargeted=result[False]["SVM-OPT"][8000],
        SVMOPT_10000_untargeted=result[False]["SVM-OPT"][10000],

        GeoDA_1000_untargeted=result[False]["GeoDA"][1000],
        GeoDA_2000_untargeted=result[False]["GeoDA"][2000],
        GeoDA_5000_untargeted=result[False]["GeoDA"][5000],
        GeoDA_8000_untargeted=result[False]["GeoDA"][8000],
        GeoDA_10000_untargeted=result[False]["GeoDA"][10000],

        Evolutionary_1000_untargeted=result[False]["Evolutionary"][1000],
        Evolutionary_2000_untargeted=result[False]["Evolutionary"][2000],
        Evolutionary_5000_untargeted=result[False]["Evolutionary"][5000],
        Evolutionary_8000_untargeted=result[False]["Evolutionary"][8000],
        Evolutionary_10000_untargeted=result[False]["Evolutionary"][10000],

        SurFree_1000_untargeted=result[False]["SurFree"][1000],
        SurFree_2000_untargeted=result[False]["SurFree"][2000],
        SurFree_5000_untargeted=result[False]["SurFree"][5000],
        SurFree_8000_untargeted=result[False]["SurFree"][8000],
        SurFree_10000_untargeted=result[False]["SurFree"][10000],

        TriangleAttack_1000_untargeted=result[False]["Triangle Attack"][1000],
        TriangleAttack_2000_untargeted=result[False]["Triangle Attack"][2000],
        TriangleAttack_5000_untargeted=result[False]["Triangle Attack"][5000],
        TriangleAttack_8000_untargeted=result[False]["Triangle Attack"][8000],
        TriangleAttack_10000_untargeted=result[False]["Triangle Attack"][10000],

        SQBA_1000_untargeted=result[False]["SQBA"][1000],
        SQBA_2000_untargeted=result[False]["SQBA"][2000],
        SQBA_5000_untargeted=result[False]["SQBA"][5000],
        SQBA_8000_untargeted=result[False]["SQBA"][8000],
        SQBA_10000_untargeted=result[False]["SQBA"][10000],

        PriorSignOPT_1prior_1000_untargeted=result[False]["Prior-Sign-OPT-1prior"][1000],
        PriorSignOPT_1prior_2000_untargeted=result[False]["Prior-Sign-OPT-1prior"][2000],
        PriorSignOPT_1prior_5000_untargeted=result[False]["Prior-Sign-OPT-1prior"][5000],
        PriorSignOPT_1prior_8000_untargeted=result[False]["Prior-Sign-OPT-1prior"][8000],
        PriorSignOPT_1prior_10000_untargeted=result[False]["Prior-Sign-OPT-1prior"][10000],

        PriorSignOPT_2prior_1000_untargeted=result[False]["Prior-Sign-OPT-2prior"][1000],
        PriorSignOPT_2prior_2000_untargeted=result[False]["Prior-Sign-OPT-2prior"][2000],
        PriorSignOPT_2prior_5000_untargeted=result[False]["Prior-Sign-OPT-2prior"][5000],
        PriorSignOPT_2prior_8000_untargeted=result[False]["Prior-Sign-OPT-2prior"][8000],
        PriorSignOPT_2prior_10000_untargeted=result[False]["Prior-Sign-OPT-2prior"][10000],

        PriorSignOPT_PGD_init_theta_1000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][1000],
        PriorSignOPT_PGD_init_theta_2000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][2000],
        PriorSignOPT_PGD_init_theta_5000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][5000],
        PriorSignOPT_PGD_init_theta_8000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][8000],
        PriorSignOPT_PGD_init_theta_10000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][10000],

        PriorOPT_1prior_1000_untargeted=result[False]["Prior-OPT-1prior"][1000],
        PriorOPT_1prior_2000_untargeted=result[False]["Prior-OPT-1prior"][2000],
        PriorOPT_1prior_5000_untargeted=result[False]["Prior-OPT-1prior"][5000],
        PriorOPT_1prior_8000_untargeted=result[False]["Prior-OPT-1prior"][8000],
        PriorOPT_1prior_10000_untargeted=result[False]["Prior-OPT-1prior"][10000],

        PriorOPT_2prior_1000_untargeted=result[False]["Prior-OPT-2prior"][1000],
        PriorOPT_2prior_2000_untargeted=result[False]["Prior-OPT-2prior"][2000],
        PriorOPT_2prior_5000_untargeted=result[False]["Prior-OPT-2prior"][5000],
        PriorOPT_2prior_8000_untargeted=result[False]["Prior-OPT-2prior"][8000],
        PriorOPT_2prior_10000_untargeted=result[False]["Prior-OPT-2prior"][10000],

        PriorOPT_PGD_init_theta_1000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][1000],
        PriorOPT_PGD_init_theta_2000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][2000],
        PriorOPT_PGD_init_theta_5000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][5000],
        PriorOPT_PGD_init_theta_8000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][8000],
        PriorOPT_PGD_init_theta_10000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][10000],

        BBA_1000_targeted=result[True]["BBA"][1000],
        BBA_2000_targeted=result[True]["BBA"][2000],
        BBA_5000_targeted=result[True]["BBA"][5000],
        BBA_8000_targeted=result[True]["BBA"][8000],
        BBA_10000_targeted=result[True]["BBA"][10000],
        BBA_15000_targeted=result[True]["BBA"][15000],
        BBA_20000_targeted=result[True]["BBA"][20000],

        HSJA_1000_targeted=result[True]["HSJA"][1000],
        HSJA_2000_targeted=result[True]["HSJA"][2000],
        HSJA_5000_targeted=result[True]["HSJA"][5000],
        HSJA_8000_targeted=result[True]["HSJA"][8000],
        HSJA_10000_targeted=result[True]["HSJA"][10000],
        HSJA_15000_targeted=result[True]["HSJA"][15000],
        HSJA_20000_targeted=result[True]["HSJA"][20000],

        TA_1000_targeted=result[True]["TA"][1000],
        TA_2000_targeted=result[True]["TA"][2000],
        TA_5000_targeted=result[True]["TA"][5000],
        TA_8000_targeted=result[True]["TA"][8000],
        TA_10000_targeted=result[True]["TA"][10000],
        TA_15000_targeted=result[True]["TA"][15000],
        TA_20000_targeted=result[True]["TA"][20000],

        GTA_1000_targeted=result[True]["G-TA"][1000],
        GTA_2000_targeted=result[True]["G-TA"][2000],
        GTA_5000_targeted=result[True]["G-TA"][5000],
        GTA_8000_targeted=result[True]["G-TA"][8000],
        GTA_10000_targeted=result[True]["G-TA"][10000],
        GTA_15000_targeted=result[True]["G-TA"][15000],
        GTA_20000_targeted=result[True]["G-TA"][20000],

        QEBA_1000_targeted=result[True]["QEBA"][1000],
        QEBA_2000_targeted=result[True]["QEBA"][2000],
        QEBA_5000_targeted=result[True]["QEBA"][5000],
        QEBA_8000_targeted=result[True]["QEBA"][8000],
        QEBA_10000_targeted=result[True]["QEBA"][10000],
        QEBA_15000_targeted=result[True]["QEBA"][15000],
        QEBA_20000_targeted=result[True]["QEBA"][20000],

        SignOPT_1000_targeted=result[True]["Sign-OPT"][1000],
        SignOPT_2000_targeted=result[True]["Sign-OPT"][2000],
        SignOPT_5000_targeted=result[True]["Sign-OPT"][5000],
        SignOPT_8000_targeted=result[True]["Sign-OPT"][8000],
        SignOPT_10000_targeted=result[True]["Sign-OPT"][10000],
        SignOPT_15000_targeted=result[True]["Sign-OPT"][15000],
        SignOPT_20000_targeted=result[True]["Sign-OPT"][20000],

        SVMOPT_1000_targeted=result[True]["SVM-OPT"][1000],
        SVMOPT_2000_targeted=result[True]["SVM-OPT"][2000],
        SVMOPT_5000_targeted=result[True]["SVM-OPT"][5000],
        SVMOPT_8000_targeted=result[True]["SVM-OPT"][8000],
        SVMOPT_10000_targeted=result[True]["SVM-OPT"][10000],
        SVMOPT_15000_targeted=result[True]["SVM-OPT"][15000],
        SVMOPT_20000_targeted=result[True]["SVM-OPT"][20000],

        GeoDA_1000_targeted=result[True]["GeoDA"][1000],
        GeoDA_2000_targeted=result[True]["GeoDA"][2000],
        GeoDA_5000_targeted=result[True]["GeoDA"][5000],
        GeoDA_8000_targeted=result[True]["GeoDA"][8000],
        GeoDA_10000_targeted=result[True]["GeoDA"][10000],
        GeoDA_15000_targeted=result[True]["GeoDA"][15000],
        GeoDA_20000_targeted=result[True]["GeoDA"][20000],

        Evolutionary_1000_targeted=result[True]["Evolutionary"][1000],
        Evolutionary_2000_targeted=result[True]["Evolutionary"][2000],
        Evolutionary_5000_targeted=result[True]["Evolutionary"][5000],
        Evolutionary_8000_targeted=result[True]["Evolutionary"][8000],
        Evolutionary_10000_targeted=result[True]["Evolutionary"][10000],
        Evolutionary_15000_targeted=result[True]["Evolutionary"][15000],
        Evolutionary_20000_targeted=result[True]["Evolutionary"][20000],

        SurFree_1000_targeted=result[True]["SurFree"][1000],
        SurFree_2000_targeted=result[True]["SurFree"][2000],
        SurFree_5000_targeted=result[True]["SurFree"][5000],
        SurFree_8000_targeted=result[True]["SurFree"][8000],
        SurFree_10000_targeted=result[True]["SurFree"][10000],
        SurFree_15000_targeted=result[True]["SurFree"][15000],
        SurFree_20000_targeted=result[True]["SurFree"][20000],

        TriangleAttack_1000_targeted=result[True]["Triangle Attack"][1000],
        TriangleAttack_2000_targeted=result[True]["Triangle Attack"][2000],
        TriangleAttack_5000_targeted=result[True]["Triangle Attack"][5000],
        TriangleAttack_8000_targeted=result[True]["Triangle Attack"][8000],
        TriangleAttack_10000_targeted=result[True]["Triangle Attack"][10000],
        TriangleAttack_15000_targeted=result[True]["Triangle Attack"][15000],
        TriangleAttack_20000_targeted=result[True]["Triangle Attack"][20000],

        SQBA_1000_targeted=result[True]["SQBA"][1000],
        SQBA_2000_targeted=result[True]["SQBA"][2000],
        SQBA_5000_targeted=result[True]["SQBA"][5000],
        SQBA_8000_targeted=result[True]["SQBA"][8000],
        SQBA_10000_targeted=result[True]["SQBA"][10000],
        SQBA_15000_targeted=result[True]["SQBA"][15000],
        SQBA_20000_targeted=result[True]["SQBA"][20000],

        PriorSignOPT_1prior_1000_targeted=result[True]["Prior-Sign-OPT-1prior"][1000],
        PriorSignOPT_1prior_2000_targeted=result[True]["Prior-Sign-OPT-1prior"][2000],
        PriorSignOPT_1prior_5000_targeted=result[True]["Prior-Sign-OPT-1prior"][5000],
        PriorSignOPT_1prior_8000_targeted=result[True]["Prior-Sign-OPT-1prior"][8000],
        PriorSignOPT_1prior_10000_targeted=result[True]["Prior-Sign-OPT-1prior"][10000],
        PriorSignOPT_1prior_15000_targeted=result[True]["Prior-Sign-OPT-1prior"][15000],
        PriorSignOPT_1prior_20000_targeted=result[True]["Prior-Sign-OPT-1prior"][20000],

        PriorSignOPT_2prior_1000_targeted=result[True]["Prior-Sign-OPT-2prior"][
            1000],
        PriorSignOPT_2prior_2000_targeted=result[True]["Prior-Sign-OPT-2prior"][
            2000],
        PriorSignOPT_2prior_5000_targeted=result[True]["Prior-Sign-OPT-2prior"][
            5000],
        PriorSignOPT_2prior_8000_targeted=result[True]["Prior-Sign-OPT-2prior"][
            8000],
        PriorSignOPT_2prior_10000_targeted=result[True]["Prior-Sign-OPT-2prior"][
            10000],
        PriorSignOPT_2prior_15000_targeted=result[True]["Prior-Sign-OPT-2prior"][
            15000],
        PriorSignOPT_2prior_20000_targeted=result[True]["Prior-Sign-OPT-2prior"][
            20000],

        PriorSignOPT_PGD_init_theta_1000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            1000],
        PriorSignOPT_PGD_init_theta_2000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            2000],
        PriorSignOPT_PGD_init_theta_5000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            5000],
        PriorSignOPT_PGD_init_theta_8000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            8000],
        PriorSignOPT_PGD_init_theta_10000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            10000],
        PriorSignOPT_PGD_init_theta_15000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            15000],
        PriorSignOPT_PGD_init_theta_20000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            20000],

        PriorOPT_1prior_1000_targeted=result[True]["Prior-OPT-1prior"][1000],
        PriorOPT_1prior_2000_targeted=result[True]["Prior-OPT-1prior"][2000],
        PriorOPT_1prior_5000_targeted=result[True]["Prior-OPT-1prior"][5000],
        PriorOPT_1prior_8000_targeted=result[True]["Prior-OPT-1prior"][8000],
        PriorOPT_1prior_10000_targeted=result[True]["Prior-OPT-1prior"][10000],
        PriorOPT_1prior_15000_targeted=result[True]["Prior-OPT-1prior"][15000],
        PriorOPT_1prior_20000_targeted=result[True]["Prior-OPT-1prior"][20000],

        PriorOPT_2prior_1000_targeted=result[True]["Prior-OPT-2prior"][1000],
        PriorOPT_2prior_2000_targeted=result[True]["Prior-OPT-2prior"][2000],
        PriorOPT_2prior_5000_targeted=result[True]["Prior-OPT-2prior"][5000],
        PriorOPT_2prior_8000_targeted=result[True]["Prior-OPT-2prior"][8000],
        PriorOPT_2prior_10000_targeted=result[True]["Prior-OPT-2prior"][10000],
        PriorOPT_2prior_15000_targeted=result[True]["Prior-OPT-2prior"][15000],
        PriorOPT_2prior_20000_targeted=result[True]["Prior-OPT-2prior"][20000],

        PriorOPT_PGD_init_theta_1000_targeted=result[True]["Prior-OPT_PGD_init_theta"][1000],
        PriorOPT_PGD_init_theta_2000_targeted=result[True]["Prior-OPT_PGD_init_theta"][2000],
        PriorOPT_PGD_init_theta_5000_targeted=result[True]["Prior-OPT_PGD_init_theta"][5000],
        PriorOPT_PGD_init_theta_8000_targeted=result[True]["Prior-OPT_PGD_init_theta"][8000],
        PriorOPT_PGD_init_theta_10000_targeted=result[True]["Prior-OPT_PGD_init_theta"][10000],
        PriorOPT_PGD_init_theta_15000_targeted=result[True]["Prior-OPT_PGD_init_theta"][15000],
        PriorOPT_PGD_init_theta_20000_targeted=result[True]["Prior-OPT_PGD_init_theta"][20000],
    )
    )


if __name__ == "__main__":
    dataset = "ImageNet"
    norm = "l2"
    if "CIFAR" in dataset:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        wanted_one_prior_list = ["inceptionresnetv2", "resnet50"]
        wanted_two_priors_list = [('xception', 'inceptionresnetv2'), ('resnet50', 'convit_base')]


    if "CIFAR" in dataset:
        targeted_result = {}
        for arch in archs:
            result = fetch_all_json_content_given_contraint(dataset, norm, True, arch, query_budgets, "mean_distortion")
            targeted_result[arch] = result
        untargeted_result = {}
        for arch in archs:
            result = fetch_all_json_content_given_contraint(dataset, norm, False, arch, query_budgets, "mean_distortion")
            untargeted_result[arch] = result

    else:
        result_archs = defaultdict(dict)
        # arch = "jx_vit"
        # surrogate_1 = "ResNet50"
        # surrogate_2 = "ConViT"
        # query_budgets = [1000, 2000, 5000, 8000, 10000]
        # result = fetch_all_json_content_given_contraint(dataset, norm, False, arch, query_budgets, wanted_one_prior_list,
        #                                                 wanted_two_priors_list, "mean_distortion")
        # result_archs[False] = result
        # query_budgets = [1000, 2000, 5000, 8000, 10000, 15000, 20000]
        # result = fetch_all_json_content_given_contraint(dataset, norm, True, arch, query_budgets,wanted_one_prior_list,
        #                                                 wanted_two_priors_list, "mean_distortion")
        # result_archs[True] = result
        #
        # draw_tables_for_ImageNet(result_archs, surrogate_1, surrogate_2)

        arch = "resnet101"
        query_budgets = [1000, 2000, 5000, 8000, 10000]
        result = fetch_json_content_for_multiple_priors(dataset, norm, False, arch, query_budgets, "mean_distortion")
        result_archs[False] = result
        arch = "swin_base_patch4_window7_224"
        result = fetch_json_content_for_multiple_priors(dataset, norm, False, arch, query_budgets, "mean_distortion")
        result_archs[True] = result
        draw_tables_for_multi_priors(result_archs, "ResNet50","ConViT","CrossViT","MaxViT","ViT")
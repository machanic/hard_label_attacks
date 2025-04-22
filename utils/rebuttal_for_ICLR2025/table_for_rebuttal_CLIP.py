import math
from collections import defaultdict
import numpy as np
import json
import os
from decimal import Decimal, ROUND_HALF_UP
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



# method_name_to_paper = {"SignOPT":"SignOPT",
#                         "PriorSignOPT":"PriorSignOPT", "PriorOPT":"PriorOPT"}
method_name_to_paper = {"SignOPT":"SignOPT","PriorOPT":"PriorOPT"}

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
    elif method == "BBA":
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
    elif method == "SurFree":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PurePriorOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PurePriorSignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPTwithPGD" or method == "PriorSignOPTwithPGD":
        path = "{method}-{dataset}-{norm}-{target_str}_with_PGD_init_theta".format(method=method[:method.index("with")], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "QEBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SQBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PDA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SQBA(PGD)":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "BBA(PGD)":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "BASES":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    return path


def read_json_and_extract(json_path):
    with (open(json_path, "r")) as file_obj:
        json_content = json.load(file_obj)
        distortion = defaultdict(dict)
        for img_id, query_distortion_dict in json_content["distortion"].items():

            distortion[int(img_id)] = query_distortion_dict
        print("read {} images for {}".format(len(distortion), json_path))
        return distortion, json_content

def read_defense_json_and_extract(json_path):
    with (open(json_path, "r")) as file_obj:
        json_content = json.load(file_obj)
        return json_content
def get_file_name_list(dataset, method_name_to_paper, norm, defense, targeted):
    folder_path_dict = {}
    for method, paper_method_name in method_name_to_paper.items():
        if method == "BASES":
            continue
        file_path = "F:/logs/hard_label_attack_complete/" + from_method_to_dir_path(dataset, method, norm, targeted)
        folder_path_dict[paper_method_name] = file_path
    return folder_path_dict

def get_mean_and_median_queries(distortion_dict, success_distortion_threshold, max_queries):
    success_dict = {}
    query_dict = {}
    min_distortion_dict = {}
    for image_index, query_distortion in distortion_dict.items():
        query_distortion = {float(query): float(dist) for query, dist in query_distortion.items() if int(float(query)) <= max_queries}
        queries = list(query_distortion.keys())
        queries = np.sort(queries)
        sorted_queries = sorted(query_distortion.items(), key=lambda x: x[1], reverse=True)
        result_query = next((query for query, dist in sorted_queries if dist <= success_distortion_threshold), None)

        find_index = np.searchsorted(queries, max_queries, side='right') - 1
        if max_queries < queries[find_index]:
            print(
                "query budget is {}, find query is {}, min query is {}, len query_distortion is {}".format(max_queries,
                                                                                                           queries[find_index],
                                                                                                           np.min(queries).item(),
                                                                                                           len(query_distortion)))
            continue

        min_distortion = query_distortion[queries[find_index]]
        min_distortion_dict[image_index] = min_distortion
        success_dict[image_index] = min_distortion <= success_distortion_threshold
        if result_query is not None:
            query_dict[image_index] = result_query
        else:
            query_dict[image_index] = max_queries  # if it does not attack successfully, use maximum queries

    mean_query = round(np.mean(list(query_dict.values())))
    median_query = round(np.median(list(query_dict.values())))
    min_distortion = round(np.mean(list(min_distortion_dict.values())), 3)
    attack_success_rate = "{}%".format(round(np.mean(list(success_dict.values())) * 100,1))
    return mean_query, median_query, min_distortion, attack_success_rate


def get_query_budgets_to_asr(distortion_dict, query_budgets, success_distortion_threshold):
    query_to_asr = {}
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
        success_list = distortion_list <= success_distortion_threshold
        success_list = success_list.astype(np.float32)
        success_rate = np.mean(success_list) * 100.0
        query_to_asr[query_budget] = "{}%".format(round(success_rate,1))
    return query_to_asr


def fetch_all_json_content_given_contraint(dataset, norm, arch, targeted, success_distortion_threshold,query_budgets,max_queries):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm,False, targeted)
    mean_query_result = defaultdict(dict)
    query_to_asr_result = defaultdict(dict)
    attack_success_rate_result = defaultdict(dict)
    surrogate_new_name_dict = {"inceptionresnetv2":"IncResV2","resnet50":"ResNet50",
                               } # FIXME #"convit_base":"ConViT"} "inceptionresnetv2":"IncResV2",
    arch_to_surrogate_arch = {}
    for method, folder in folder_list.items():
        for file_name in os.listdir(folder):
            new_method = None
            if file_name.startswith(arch) and file_name.endswith(".json"):
                file_path = folder + "/"+file_name
                if targeted:
                    if method in ["Triangle Attack", "RayS", "GeoDA"]:
                        mean_query_result[arch] = defaultdict(lambda: "-")
                        query_to_asr_result[arch] = defaultdict(lambda: "-")
                        continue
                if norm == "l2":
                    if "RayS" == method:
                        mean_query_result[arch] = defaultdict(lambda: "-")
                        query_to_asr_result[arch] = defaultdict(lambda: "-")
                        continue
                elif norm == "linf":
                    if method in ["Evolutionary","SurFree","Triangle Attack","RayS"]:
                        mean_query_result[arch] = defaultdict(lambda: "-")
                        query_to_asr_result[arch] = defaultdict(lambda: "-")
                        continue

                distortion_dict, json_content = read_json_and_extract(file_path)
                if "PGD_init_theta" in json_content["args"]:
                    PGD_init = json_content["args"]["PGD_init_theta"]
                else:
                    PGD_init = False
                if "surrogate_arch" in json_content["args"] or "surrogate_archs" in json_content["args"]:
                    if json_content["args"]["surrogate_arch"] is not None:    # PriorSignOPTwithPGDIncResV2
                        if json_content["args"]["surrogate_arch"] not in surrogate_new_name_dict:
                            print("skip {}".format(file_path))
                            continue
                        if PGD_init:
                            new_method = method +  surrogate_new_name_dict[
                                json_content["args"]["surrogate_arch"]]
                            arch_to_surrogate_arch[arch] = surrogate_new_name_dict[
                                json_content["args"]["surrogate_arch"]]
                        else:
                            new_method = method + "with" + surrogate_new_name_dict[
                                json_content["args"]["surrogate_arch"]]
                            arch_to_surrogate_arch[arch] = surrogate_new_name_dict[
                                json_content["args"]["surrogate_arch"]]
                        if "surrogate_archs" in json_content["args"]:
                            assert json_content["args"]["surrogate_archs"] is None
                    elif "surrogate_archs" in json_content["args"] and json_content["args"]["surrogate_archs"] is not None:
                        if len(json_content["args"]["surrogate_archs"]) == 1:
                            if json_content["args"]["surrogate_archs"][0] not in surrogate_new_name_dict:
                                print("skip {}".format(file_path))
                                continue
                            if PGD_init:
                                new_method = method + surrogate_new_name_dict[
                                    json_content["args"]["surrogate_archs"][0]]
                                arch_to_surrogate_arch[arch] = surrogate_new_name_dict[
                                    json_content["args"]["surrogate_archs"][0]]
                            else:
                                new_method = method + "with" + surrogate_new_name_dict[
                                    json_content["args"]["surrogate_archs"][0]]
                                arch_to_surrogate_arch[arch] = surrogate_new_name_dict[
                                    json_content["args"]["surrogate_archs"][0]]
                        else:
                            print("skip {}".format(file_name))
                            continue
                mean_query, median_query,min_distortion, attack_success_rate = get_mean_and_median_queries(distortion_dict, success_distortion_threshold, max_queries)
                query_to_asr = get_query_budgets_to_asr(distortion_dict, query_budgets, success_distortion_threshold)
                if new_method:
                    if "SQBA" in new_method or "BBA" in new_method:
                        new_method = new_method[:new_method.index("with")]  # for mean queries
                    mean_query_result[arch][new_method] = mean_query
                    query_to_asr_result[arch][new_method] = query_to_asr
                    attack_success_rate_result[arch][new_method] = attack_success_rate
                else:
                    mean_query_result[arch][method] = mean_query
                    query_to_asr_result[arch][method] = query_to_asr
                    attack_success_rate_result[arch][method] = attack_success_rate
    # for arch in query_to_asr_result:
    #     print(arch + '  -------------------')
    #     for method in query_to_asr_result[arch]:
    #         print(method)
    return mean_query_result, query_to_asr_result, attack_success_rate_result, arch_to_surrogate_arch


def get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets, want_key):
    mean_and_median_distortions = {}
    for query_budget in query_budgets:
        distortion_list = []
        for image_index, query_distortion in distortion_dict.items():
            query_distortion = {float(query):float(dist) for query,dist in query_distortion.items()}
            queries = list(query_distortion.keys())
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

def fetch_all_json_content_for_distortion(dataset, norm, arch, targeted, query_budgets):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm,False, targeted)
    mean_distortion_result = defaultdict(dict)

    surrogate_new_name_dict = { "inceptionresnetv2": "IncResV2"
                               }
    order_list = ["resnet50", "convit_base", "crossvit_base_224", "maxvit_rmlp_small_rw_224", "jx_vit","inceptionresnetv2","xception"]
    order_list = [surrogate_new_name_dict[key] for key in order_list if key in surrogate_new_name_dict]
    arch_to_surrogate_arch = defaultdict(set)
    for method, folder in folder_list.items():
        for file_name in os.listdir(folder):
            new_method = None
            if file_name.startswith(arch) and file_name.endswith(".json"):
                file_path = folder + "/"+file_name
                if targeted:
                    if method in ["Triangle Attack", "RayS", "GeoDA"]:
                        mean_distortion_result[arch] = defaultdict(lambda: "-")
                        continue
                if norm == "l2":
                    if "RayS" == method:
                        mean_distortion_result[arch] = defaultdict(lambda: "-")
                        continue
                elif norm == "linf":
                    if method in ["Evolutionary","SurFree","Triangle Attack","RayS"]:
                        mean_distortion_result[arch] = defaultdict(lambda: "-")
                        continue

                distortion_dict, json_content = read_json_and_extract(file_path)
                if "PGD_init_theta" in json_content["args"]:
                    PGD_init = json_content["args"]["PGD_init_theta"]
                else:
                    PGD_init = False
                if "surrogate_arch" in json_content["args"] or "surrogate_archs" in json_content["args"]:
                    if json_content["args"]["surrogate_arch"] is not None:    # PriorSignOPTwithPGDIncResV2
                        if json_content["args"]["surrogate_arch"] not in surrogate_new_name_dict:
                            print("skip {}".format(file_path))
                            continue
                        if PGD_init:
                            new_method = method + surrogate_new_name_dict[
                                json_content["args"]["surrogate_arch"]]

                            arch_to_surrogate_arch[arch].add(surrogate_new_name_dict[
                                    json_content["args"]["surrogate_arch"]])
                        else:
                            new_method = method + "with" + surrogate_new_name_dict[
                                json_content["args"]["surrogate_arch"]]

                            arch_to_surrogate_arch[arch].add(surrogate_new_name_dict[
                                json_content["args"]["surrogate_arch"]])
                        if "surrogate_archs" in json_content["args"]:
                            assert json_content["args"]["surrogate_archs"] is None
                    elif "surrogate_archs" in json_content["args"] and json_content["args"]["surrogate_archs"] is not None:
                        must_skip = False
                        for surrogate_arch in json_content["args"]["surrogate_archs"]:
                            if surrogate_arch not in surrogate_new_name_dict:
                                must_skip = True
                        if must_skip:
                            print("skip {}".format(file_path))
                            continue
                        surrogate_model_names = ",".join(surrogate_new_name_dict[surrogate_arch] for surrogate_arch in json_content["args"]["surrogate_archs"])
                        if PGD_init:
                            new_method = method + surrogate_model_names
                        else:
                            new_method = method + "with"+ surrogate_model_names
                        arch_to_surrogate_arch[arch].update(set(surrogate_model_names.split(',')))
                print("reading "+file_path)
                mean_distortion = get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets, "mean_distortion")
                if new_method:
                    if "SQBA" in new_method or "BBA" in new_method:
                        new_method = new_method[:new_method.index("with")]  # for mean queries
                    mean_distortion_result[arch][new_method] = mean_distortion
                else:
                    mean_distortion_result[arch][method] = mean_distortion
    new_arch_to_surrogate_arch = {}
    for arch in arch_to_surrogate_arch.keys():
        arch_set = arch_to_surrogate_arch[arch]
        new_arch_to_surrogate_arch[arch] = [word for word in order_list if word in arch_set]
    return mean_distortion_result, new_arch_to_surrogate_arch


def is_continuous_subsequence(list_a, list_b):
    """
    判断 list_a 是否是 list_b 的连续子序列

    Args:
        list_a (list): 可能是子序列的列表
        list_b (list): 父列表

    Returns:
        bool: 如果 list_a 是 list_b 的连续子序列，返回 True；否则返回 False
    """
    n, m = len(list_a), len(list_b)
    if n == 0:  # 空列表是任何列表的连续子序列
        return False
    if n > m:  # 子序列长度大于母列表，不可能是子序列
        return False

    # 遍历 list_b，检查是否存在和 list_a 匹配的连续子序列
    for i in range(m - n + 1):
        if list_b[i:i + n] == list_a:
            return True
    return False

def fetch_all_json_content_for_comparing_priors_ASR_mean_queries(dataset, norm, arch, targeted, success_distortion_threshold,max_queries):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm,False, targeted)
    nested_dict = lambda: defaultdict(lambda: "-")
    all_result = defaultdict(nested_dict)
    order_list = ["resnet50", "convit_base", "crossvit_base_224", "maxvit_rmlp_small_rw_224", "jx_vit"]
    for method, folder in folder_list.items():
        for file_name in os.listdir(folder):
            new_method = None
            if file_name.startswith(arch) and file_name.endswith(".json"):
                file_path = folder + "/"+file_name
                distortion_dict, json_content = read_json_and_extract(file_path)
                if "PGD_init_theta" in json_content["args"]:
                    PGD_init = json_content["args"]["PGD_init_theta"]
                else:
                    PGD_init = False
                if "surrogate_arch" in json_content["args"] or "surrogate_archs" in json_content["args"]:
                    if json_content["args"]["surrogate_arch"] is not None:
                        if json_content["args"]["surrogate_arch"] not in order_list:
                            print("skip {}".format(file_path))
                            continue
                        if PGD_init:
                            new_method = method + "1"
                        else:
                            new_method = method + "1"
                        if "surrogate_archs" in json_content["args"]:
                            assert json_content["args"]["surrogate_archs"] is None
                    elif "surrogate_archs" in json_content["args"] and json_content["args"]["surrogate_archs"] is not None:
                        if not is_continuous_subsequence(json_content["args"]["surrogate_archs"], order_list):
                            print("skip {}".format(file_path))
                            continue
                        # if len(json_content["args"]["surrogate_archs"]) == 1 and json_content["args"]["surrogate_archs"][0] == "resnet50":
                        #     print("skip {}".format(file_path))
                        #     continue
                        if PGD_init:
                            new_method = method + str(len(json_content["args"]["surrogate_archs"]))
                        else:
                            new_method = method + str(len(json_content["args"]["surrogate_archs"]))

                mean_query, median_query, min_distortion, attack_success_rate = get_mean_and_median_queries(distortion_dict, success_distortion_threshold, max_queries)
                if new_method:
                    all_result[new_method]["ASR"] = attack_success_rate
                    all_result[new_method]["distortion"] = min_distortion
                    all_result[new_method]["queries"] = mean_query
                else:
                    all_result[method]["ASR"] = attack_success_rate
                    all_result[method]["distortion"] = min_distortion
                    all_result[method]["queries"] = mean_query

    return all_result

def draw_distortion_table(result, surrogate_arch):
    print(r"""
            | Method | @1K | @3K | @5K | @7K |  @8K |  @10K |
            | :- | :- |:- | :- | :- | :- | :- |
            | HSJA | {HSJA_1K} |{HSJA_3K} | {HSJA_5K} | {HSJA_7K} | {HSJA_8K} | {HSJA_10K} |
            | TA | {TA_1K} |{TA_3K} | {TA_5K} | {TA_7K} | {TA_8K} | {TA_10K} |
            | G-TA | {GTA_1K} |{GTA_3K} | {GTA_5K} | {GTA_7K} | {GTA_8K} | {GTA_10K} |
            | GeoDA | {GeoDA_1K} |{GeoDA_3K} | {GeoDA_5K} | {GeoDA_7K} | {GeoDA_8K} | {GeoDA_10K} |
            | Evolutionary | {Evolutionary_1K} |{Evolutionary_3K} | {Evolutionary_5K} | {Evolutionary_7K} | {Evolutionary_8K} | {Evolutionary_10K} |
            | SurFree | {SurFree_1K} |{SurFree_3K} | {SurFree_5K} | {SurFree_7K} | {SurFree_8K} | {SurFree_10K} |
            | Triangle Attack | {TriA_1K} |{TriA_3K} | {TriA_5K} | {TriA_7K} | {TriA_8K} | {TriA_10K} |
            | SVM-OPT | {SVMOPT_1K} |{SVMOPT_3K} | {SVMOPT_5K} | {SVMOPT_7K} | {SVMOPT_8K} | {SVMOPT_10K} |
            | Sign-OPT  | {SignOPT_1K} |{SignOPT_3K} | {SignOPT_5K} | {SignOPT_7K} | {SignOPT_8K} | {SignOPT_10K} |
            | $\text{{SQBA}}_\text{{{surrogate_arch}}}$ | {SQBA_1K} |{SQBA_3K} | {SQBA_5K} | {SQBA_7K} | {SQBA_8K} | {SQBA_10K} |
            | $\text{{BBA}}_\text{{{surrogate_arch}}}$ | {BBA_1K} |{BBA_3K} | {BBA_5K} | {BBA_7K} | {BBA_8K} | {BBA_10K} |
            | $\text{{Pure-Prior-Sign-OPT}}_\text{{{surrogate_arch}}}$ | {PurePriorSignOPT_1K} | {PurePriorSignOPT_3K} | {PurePriorSignOPT_5K} | {PurePriorSignOPT_7K} | {PurePriorSignOPT_8K} | {PurePriorSignOPT_10K} |
            | $\text{{Pure-Prior-OPT}}_\text{{{surrogate_arch}}}$ | {PurePriorOPT_1K} | {PurePriorOPT_3K} | {PurePriorOPT_5K} | {PurePriorOPT_7K} | {PurePriorOPT_8K} | {PurePriorOPT_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{{surrogate_arch}}}$ | {PriorSignOPT_1K} | {PriorSignOPT_3K} | {PriorSignOPT_5K} | {PriorSignOPT_7K} | {PriorSignOPT_8K} | {PriorSignOPT_10K} |
            | $\text{{Prior-OPT}} _ \text{{{surrogate_arch}}}$ | {PriorOPT_1K} | {PriorOPT_3K} | {PriorOPT_5K} |  {PriorOPT_7K} | {PriorOPT_8K} | {PriorOPT_10K} |
                                """.format(
        surrogate_arch=surrogate_arch,
        HSJA_1K=result["HSJA"][1000],
        HSJA_3K=result["HSJA"][3000],
        HSJA_5K=result["HSJA"][5000],
        HSJA_7K=result["HSJA"][7000],
        HSJA_8K=result["HSJA"][8000],
        HSJA_10K=result["HSJA"][10000],

        TA_1K=result["TA"][1000],
        TA_3K=result["TA"][3000],
        TA_5K=result["TA"][5000],
        TA_7K=result["TA"][7000],
        TA_8K=result["TA"][8000],
        TA_10K=result["TA"][10000],

        GTA_1K=result["G-TA"][1000],
        GTA_3K=result["G-TA"][3000],
        GTA_5K=result["G-TA"][5000],
        GTA_7K=result["G-TA"][7000],
        GTA_8K=result["G-TA"][8000],
        GTA_10K=result["G-TA"][10000],

        GeoDA_1K=result["GeoDA"][1000],
        GeoDA_3K=result["GeoDA"][3000],
        GeoDA_5K=result["GeoDA"][5000],
        GeoDA_7K=result["GeoDA"][7000],
        GeoDA_8K=result["GeoDA"][8000],
        GeoDA_10K=result["GeoDA"][10000],

        Evolutionary_1K=result["Evolutionary"][1000],
        Evolutionary_3K=result["Evolutionary"][3000],
        Evolutionary_5K=result["Evolutionary"][5000],
        Evolutionary_7K=result["Evolutionary"][7000],
        Evolutionary_8K=result["Evolutionary"][8000],
        Evolutionary_10K=result["Evolutionary"][10000],

        SurFree_1K=result["SurFree"][1000],
        SurFree_3K=result["SurFree"][3000],
        SurFree_5K=result["SurFree"][5000],
        SurFree_7K=result["SurFree"][7000],
        SurFree_8K=result["SurFree"][8000],
        SurFree_10K=result["SurFree"][10000],

        TriA_1K=result["Triangle Attack"][1000],
        TriA_3K=result["Triangle Attack"][3000],
        TriA_5K=result["Triangle Attack"][5000],
        TriA_7K=result["Triangle Attack"][7000],
        TriA_8K=result["Triangle Attack"][8000],
        TriA_10K=result["Triangle Attack"][10000],


        SQBA_1K=result["SQBA"][1000],
        SQBA_3K=result["SQBA"][3000],
        SQBA_5K=result["SQBA"][5000],
        SQBA_7K=result["SQBA"][7000],
        SQBA_8K=result["SQBA"][8000],
        SQBA_10K=result["SQBA"][10000],

        BBA_1K=result["BBA"][1000],
        BBA_3K=result["BBA"][3000],
        BBA_5K=result["BBA"][5000],
        BBA_7K=result["BBA"][7000],
        BBA_8K=result["BBA"][8000],
        BBA_10K=result["BBA"][10000],

        SignOPT_1K=result["Sign-OPT"][1000],
        SignOPT_3K=result["Sign-OPT"][3000],
        SignOPT_5K=result["Sign-OPT"][5000],
        SignOPT_7K=result["Sign-OPT"][7000],
        SignOPT_8K=result["Sign-OPT"][8000],
        SignOPT_10K=result["Sign-OPT"][10000],

        SVMOPT_1K=result["SVM-OPT"][1000],
        SVMOPT_3K=result["SVM-OPT"][3000],
        SVMOPT_5K=result["SVM-OPT"][5000],
        SVMOPT_7K=result["SVM-OPT"][7000],
        SVMOPT_8K=result["SVM-OPT"][8000],
        SVMOPT_10K=result["SVM-OPT"][10000],


        PriorSignOPT_1K=result["PriorSignOPTwith"+surrogate_arch][1000],
        PriorSignOPT_3K=result["PriorSignOPTwith"+surrogate_arch][3000],
        PriorSignOPT_5K=result["PriorSignOPTwith"+surrogate_arch][5000],
        PriorSignOPT_7K=result["PriorSignOPTwith"+surrogate_arch][7000],
        PriorSignOPT_8K=result["PriorSignOPTwith"+surrogate_arch][8000],
        PriorSignOPT_10K=result["PriorSignOPTwith"+surrogate_arch][10000],


        PriorOPT_1K=result["PriorOPTwith"+surrogate_arch][1000],
        PriorOPT_3K=result["PriorOPTwith"+surrogate_arch][3000],
        PriorOPT_5K=result["PriorOPTwith"+surrogate_arch][5000],
        PriorOPT_7K=result["PriorOPTwith"+surrogate_arch][7000],
        PriorOPT_8K=result["PriorOPTwith"+surrogate_arch][8000],
        PriorOPT_10K=result["PriorOPTwith"+surrogate_arch][10000],

        PurePriorSignOPT_1K=result["PurePriorSignOPTwith"+surrogate_arch][1000],
        PurePriorSignOPT_3K=result["PurePriorSignOPTwith"+surrogate_arch][3000],
        PurePriorSignOPT_5K=result["PurePriorSignOPTwith"+surrogate_arch][5000],
        PurePriorSignOPT_7K=result["PurePriorSignOPTwith"+surrogate_arch][7000],
        PurePriorSignOPT_8K=result["PurePriorSignOPTwith"+surrogate_arch][8000],
        PurePriorSignOPT_10K=result["PurePriorSignOPTwith"+surrogate_arch][10000],


        PurePriorOPT_1K=result["PurePriorOPTwith"+surrogate_arch][1000],
        PurePriorOPT_3K=result["PurePriorOPTwith"+surrogate_arch][3000],
        PurePriorOPT_5K=result["PurePriorOPTwith"+surrogate_arch][5000],
        PurePriorOPT_7K=result["PurePriorOPTwith"+surrogate_arch][7000],
        PurePriorOPT_8K=result["PurePriorOPTwith"+surrogate_arch][8000],
        PurePriorOPT_10K=result["PurePriorOPTwith"+surrogate_arch][10000],

    )
    )


def draw_distortion_PGD_untargeted_attack(result,  surrogate_arch):
    print(r"""
            | Method | @1K | @3K | @5K | @7K |  @8K |  @10K |
            | :- | :- |:- | :- | :- | :- | :- |
            | $\text{{SQBA}}_\text{{{surrogate_arch}}}$ | {SQBA_1K} |{SQBA_3K} | {SQBA_5K} | {SQBA_7K} | {SQBA_8K} | {SQBA_10K} |
            | $\text{{SQBA(PGD)}}_\text{{{surrogate_arch}}}$ | {SQBA_PGD_1K} |{SQBA_PGD_3K} | {SQBA_PGD_5K} | {SQBA_PGD_7K} | {SQBA_PGD_8K} | {SQBA_PGD_10K} |
            | $\text{{BBA}}_\text{{{surrogate_arch}}}$ | {BBA_1K} |{BBA_3K} | {BBA_5K} | {BBA_7K} | {BBA_8K} | {BBA_10K} |
            | $\text{{BBA(PGD)}}_\text{{{surrogate_arch}}}$ | {BBA_PGD_1K} |{BBA_PGD_3K} | {BBA_PGD_5K} | {BBA_PGD_7K} | {BBA_PGD_8K} | {BBA_PGD_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{{surrogate_arch}}}$ | {PriorSignOPT_1K} | {PriorSignOPT_3K} | {PriorSignOPT_5K} | {PriorSignOPT_7K} | {PriorSignOPT_8K} | {PriorSignOPT_10K} |
            | $\text{{Prior-Sign-OPT(PGD)}}_\text{{{surrogate_arch}}}$ | {PriorSignOPTwithPGD_1K} | {PriorSignOPTwithPGD_3K} | {PriorSignOPTwithPGD_5K} | {PriorSignOPTwithPGD_7K} | {PriorSignOPTwithPGD_8K} | {PriorSignOPTwithPGD_10K} |
            | $\text{{Prior-OPT}} _ \text{{{surrogate_arch}}}$ | {PriorOPT_1K} | {PriorOPT_3K} | {PriorOPT_5K} |  {PriorOPT_7K} | {PriorOPT_8K} | {PriorOPT_10K} |
            | $\text{{Prior-OPT(PGD)}} _ \text{{{surrogate_arch}}}$ | {PriorOPTwithPGD_1K} | {PriorOPTwithPGD_3K} | {PriorOPTwithPGD_5K} |  {PriorOPTwithPGD_7K} | {PriorOPTwithPGD_8K} | {PriorOPTwithPGD_10K} |
                                """.format(
        surrogate_arch=surrogate_arch,

        SQBA_1K=result["SQBA"][1000],
        SQBA_3K=result["SQBA"][3000],
        SQBA_5K=result["SQBA"][5000],
        SQBA_7K=result["SQBA"][7000],
        SQBA_8K=result["SQBA"][8000],
        SQBA_10K=result["SQBA"][10000],

        SQBA_PGD_1K=result["SQBA(PGD)"][1000],
        SQBA_PGD_3K=result["SQBA(PGD)"][3000],
        SQBA_PGD_5K=result["SQBA(PGD)"][5000],
        SQBA_PGD_7K=result["SQBA(PGD)"][7000],
        SQBA_PGD_8K=result["SQBA(PGD)"][8000],
        SQBA_PGD_10K=result["SQBA(PGD)"][10000],

        BBA_1K=result["BBA"][1000],
        BBA_3K=result["BBA"][3000],
        BBA_5K=result["BBA"][5000],
        BBA_7K=result["BBA"][7000],
        BBA_8K=result["BBA"][8000],
        BBA_10K=result["BBA"][10000],

        BBA_PGD_1K=result["BBA(PGD)"][1000],
        BBA_PGD_3K=result["BBA(PGD)"][3000],
        BBA_PGD_5K=result["BBA(PGD)"][5000],
        BBA_PGD_7K=result["BBA(PGD)"][7000],
        BBA_PGD_8K=result["BBA(PGD)"][8000],
        BBA_PGD_10K=result["BBA(PGD)"][10000],

        PriorSignOPT_1K=result["PriorSignOPTwith" + surrogate_arch][1000],
        PriorSignOPT_3K=result["PriorSignOPTwith" + surrogate_arch][3000],
        PriorSignOPT_5K=result["PriorSignOPTwith" + surrogate_arch][5000],
        PriorSignOPT_7K=result["PriorSignOPTwith" + surrogate_arch][7000],
        PriorSignOPT_8K=result["PriorSignOPTwith" + surrogate_arch][8000],
        PriorSignOPT_10K=result["PriorSignOPTwith" + surrogate_arch][10000],

        PriorOPT_1K=result["PriorOPTwith" + surrogate_arch][1000],
        PriorOPT_3K=result["PriorOPTwith" + surrogate_arch][3000],
        PriorOPT_5K=result["PriorOPTwith" + surrogate_arch][5000],
        PriorOPT_7K=result["PriorOPTwith" + surrogate_arch][7000],
        PriorOPT_8K=result["PriorOPTwith" + surrogate_arch][8000],
        PriorOPT_10K=result["PriorOPTwith" + surrogate_arch][10000],


        PriorSignOPTwithPGD_1K=result["PriorSignOPTwithPGD" + surrogate_arch][1000],
        PriorSignOPTwithPGD_3K=result["PriorSignOPTwithPGD" + surrogate_arch][3000],
        PriorSignOPTwithPGD_5K=result["PriorSignOPTwithPGD" + surrogate_arch][5000],
        PriorSignOPTwithPGD_7K=result["PriorSignOPTwithPGD" + surrogate_arch][7000],
        PriorSignOPTwithPGD_8K=result["PriorSignOPTwithPGD" + surrogate_arch][8000],
        PriorSignOPTwithPGD_10K=result["PriorSignOPTwithPGD" + surrogate_arch][10000],

        PriorOPTwithPGD_1K=result["PriorOPTwithPGD" + surrogate_arch][1000],
        PriorOPTwithPGD_3K=result["PriorOPTwithPGD" + surrogate_arch][3000],
        PriorOPTwithPGD_5K=result["PriorOPTwithPGD" + surrogate_arch][5000],
        PriorOPTwithPGD_7K=result["PriorOPTwithPGD" + surrogate_arch][7000],
        PriorOPTwithPGD_8K=result["PriorOPTwithPGD" + surrogate_arch][8000],
        PriorOPTwithPGD_10K=result["PriorOPTwithPGD" + surrogate_arch][10000],

    )
    )


def draw_distortion_table_for_pure_priors(result, surrogate_arch1, surrogate_arch2, surrogate_arch3, surrogate_arch4,
                                          surrogate_arch5):
    print(r"""
            | Method | @1K | @3K | @5K | @7K |  @8K |  @10K |
            | :- | :- |:- | :- | :- | :- | :- |
            | Sign-OPT  | {SignOPT_1K} |{SignOPT_3K} | {SignOPT_5K} | {SignOPT_7K} | {SignOPT_8K} | {SignOPT_10K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1}}}$ | {PurePriorSignOPT1_1K} | {PurePriorSignOPT1_3K} | {PurePriorSignOPT1_5K} | {PurePriorSignOPT1_7K} | {PurePriorSignOPT1_8K} | {PurePriorSignOPT1_10K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2}}}$ | {PurePriorSignOPT2_1K} | {PurePriorSignOPT2_3K} | {PurePriorSignOPT2_5K} | {PurePriorSignOPT2_7K} | {PurePriorSignOPT2_8K} | {PurePriorSignOPT2_10K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3}}}$ | {PurePriorSignOPT3_1K} | {PurePriorSignOPT3_3K} | {PurePriorSignOPT3_5K} | {PurePriorSignOPT3_7K} | {PurePriorSignOPT3_8K} | {PurePriorSignOPT3_10K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4}}}$ | {PurePriorSignOPT4_1K} | {PurePriorSignOPT4_3K} | {PurePriorSignOPT4_5K} | {PurePriorSignOPT4_7K} | {PurePriorSignOPT4_8K} | {PurePriorSignOPT4_10K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4},{surrogate_arch5}}}$ | {PurePriorSignOPT5_1K} | {PurePriorSignOPT5_3K} | {PurePriorSignOPT5_5K} | {PurePriorSignOPT5_7K} | {PurePriorSignOPT5_8K} | {PurePriorSignOPT5_10K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1}}}$ | {PurePriorOPT1_1K} | {PurePriorOPT1_3K} | {PurePriorOPT1_5K} | {PurePriorOPT1_7K} | {PurePriorOPT1_8K} | {PurePriorOPT1_10K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2}}}$ | {PurePriorOPT2_1K} | {PurePriorOPT2_3K} | {PurePriorOPT2_5K} | {PurePriorOPT2_7K} | {PurePriorOPT2_8K} | {PurePriorOPT2_10K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3}}}$ | {PurePriorOPT3_1K} | {PurePriorOPT3_3K} | {PurePriorOPT3_5K} | {PurePriorOPT3_7K} | {PurePriorOPT3_8K} | {PurePriorOPT3_10K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4}}}$ | {PurePriorOPT4_1K} | {PurePriorOPT4_3K} | {PurePriorOPT4_5K} | {PurePriorOPT4_7K} | {PurePriorOPT4_8K} | {PurePriorOPT4_10K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4},{surrogate_arch5}}}$ | {PurePriorOPT5_1K} | {PurePriorOPT5_3K} | {PurePriorOPT5_5K} | {PurePriorOPT5_7K} | {PurePriorOPT5_8K} | {PurePriorOPT5_10K} |
            | $\text{{Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4},{surrogate_arch5}}}$ | {PriorSignOPT5_1K} | {PriorSignOPT5_3K} | {PriorSignOPT5_5K} | {PriorSignOPT5_7K} | {PriorSignOPT5_8K} | {PriorSignOPT5_10K} |
            | $\text{{Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4},{surrogate_arch5}}}$ | {PriorOPT5_1K} | {PriorOPT5_3K} | {PriorOPT5_5K} |  {PriorOPT5_7K} | {PriorOPT5_8K} | {PriorOPT5_10K} |
             """.format(
        surrogate_arch1=surrogate_arch1,
        surrogate_arch2=surrogate_arch2,
        surrogate_arch3=surrogate_arch3,
        surrogate_arch4=surrogate_arch4,
        surrogate_arch5=surrogate_arch5,

        SignOPT_1K=result["SignOPT"][1000],
        SignOPT_3K=result["SignOPT"][3000],
        SignOPT_5K=result["SignOPT"][5000],
        SignOPT_7K=result["SignOPT"][7000],
        SignOPT_8K=result["SignOPT"][8000],
        SignOPT_10K=result["SignOPT"][10000],

        PriorSignOPT5_1K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            1000],
        PriorSignOPT5_3K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            3000],
        PriorSignOPT5_5K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            5000],
        PriorSignOPT5_7K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            7000],
        PriorSignOPT5_8K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            8000],
        PriorSignOPT5_10K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            10000],

        PriorOPT5_1K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            1000],
        PriorOPT5_3K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            3000],
        PriorOPT5_5K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            5000],
        PriorOPT5_7K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            7000],
        PriorOPT5_8K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            8000],
        PriorOPT5_10K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
                10000],

        PurePriorSignOPT1_1K=result["PurePriorSignOPTwith" + surrogate_arch1][1000],
        PurePriorSignOPT1_3K=result["PurePriorSignOPTwith" + surrogate_arch1][3000],
        PurePriorSignOPT1_5K=result["PurePriorSignOPTwith" + surrogate_arch1][5000],
        PurePriorSignOPT1_7K=result["PurePriorSignOPTwith" + surrogate_arch1][7000],
        PurePriorSignOPT1_8K=result["PurePriorSignOPTwith" + surrogate_arch1][8000],
        PurePriorSignOPT1_10K=result["PurePriorSignOPTwith" + surrogate_arch1][10000],

        PurePriorSignOPT2_1K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][1000],
        PurePriorSignOPT2_3K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][3000],
        PurePriorSignOPT2_5K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][5000],
        PurePriorSignOPT2_7K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][7000],
        PurePriorSignOPT2_8K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][8000],
        PurePriorSignOPT2_10K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][10000],

        PurePriorSignOPT3_1K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][1000],
        PurePriorSignOPT3_3K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][3000],
        PurePriorSignOPT3_5K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][5000],
        PurePriorSignOPT3_7K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][7000],
        PurePriorSignOPT3_8K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][8000],
        PurePriorSignOPT3_10K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][10000],

        PurePriorSignOPT4_1K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            1000],
        PurePriorSignOPT4_3K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            3000],
        PurePriorSignOPT4_5K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            5000],
        PurePriorSignOPT4_7K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            7000],
        PurePriorSignOPT4_8K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            8000],
        PurePriorSignOPT4_10K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            10000],

        PurePriorSignOPT5_1K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            1000],
        PurePriorSignOPT5_3K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            3000],
        PurePriorSignOPT5_5K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            5000],
        PurePriorSignOPT5_7K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            7000],
        PurePriorSignOPT5_8K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            8000],
        PurePriorSignOPT5_10K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            10000],

        PurePriorOPT1_1K=result["PurePriorOPTwith" + surrogate_arch1][1000],
        PurePriorOPT1_3K=result["PurePriorOPTwith" + surrogate_arch1][3000],
        PurePriorOPT1_5K=result["PurePriorOPTwith" + surrogate_arch1][5000],
        PurePriorOPT1_7K=result["PurePriorOPTwith" + surrogate_arch1][7000],
        PurePriorOPT1_8K=result["PurePriorOPTwith" + surrogate_arch1][8000],
        PurePriorOPT1_10K=result["PurePriorOPTwith" + surrogate_arch1][10000],

        PurePriorOPT2_1K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][1000],
        PurePriorOPT2_3K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][3000],
        PurePriorOPT2_5K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][5000],
        PurePriorOPT2_7K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][7000],
        PurePriorOPT2_8K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][8000],
        PurePriorOPT2_10K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][10000],

        PurePriorOPT3_1K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][
            1000],
        PurePriorOPT3_3K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][
            3000],
        PurePriorOPT3_5K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][
            5000],
        PurePriorOPT3_7K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][
            7000],
        PurePriorOPT3_8K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][
            8000],
        PurePriorOPT3_10K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][
            10000],

        PurePriorOPT4_1K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            1000],
        PurePriorOPT4_3K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            3000],
        PurePriorOPT4_5K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            5000],
        PurePriorOPT4_7K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            7000],
        PurePriorOPT4_8K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            8000],
        PurePriorOPT4_10K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            10000],

        PurePriorOPT5_1K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            1000],
        PurePriorOPT5_3K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            3000],
        PurePriorOPT5_5K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            5000],
        PurePriorOPT5_7K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            7000],
        PurePriorOPT5_8K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            8000],
        PurePriorOPT5_10K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            10000],
    ))

def draw_distortion_targeted_attack_for_pure_priors(result, surrogate_arch1, surrogate_arch2, surrogate_arch3, surrogate_arch4,
                                          surrogate_arch5):
    print(r"""
            | Method | @1K | @3K | @5K | @7K |  @8K |  @10K | @12K | @15K | @18K | @20K |
            | :- | :- |:- | :- | :- | :- | :- | :- | :- | :- | :- |
            | Sign-OPT  | {SignOPT_1K} |{SignOPT_3K} | {SignOPT_5K} | {SignOPT_7K} | {SignOPT_8K} | {SignOPT_10K} | {SignOPT_12K} | {SignOPT_15K} | {SignOPT_18K} | {SignOPT_20K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1}}}$ | {PurePriorSignOPT1_1K} | {PurePriorSignOPT1_3K} | {PurePriorSignOPT1_5K} | {PurePriorSignOPT1_7K} | {PurePriorSignOPT1_8K} | {PurePriorSignOPT1_10K} |  {PurePriorSignOPT1_12K} |  {PurePriorSignOPT1_15K} |  {PurePriorSignOPT1_18K} |  {PurePriorSignOPT1_20K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2}}}$ | {PurePriorSignOPT2_1K} | {PurePriorSignOPT2_3K} | {PurePriorSignOPT2_5K} | {PurePriorSignOPT2_7K} | {PurePriorSignOPT2_8K} | {PurePriorSignOPT2_10K} | {PurePriorSignOPT2_12K} |  {PurePriorSignOPT2_15K} |  {PurePriorSignOPT2_18K} |  {PurePriorSignOPT2_20K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3}}}$ | {PurePriorSignOPT3_1K} | {PurePriorSignOPT3_3K} | {PurePriorSignOPT3_5K} | {PurePriorSignOPT3_7K} | {PurePriorSignOPT3_8K} | {PurePriorSignOPT3_10K} |  {PurePriorSignOPT3_12K} |  {PurePriorSignOPT3_15K} |  {PurePriorSignOPT3_18K} |  {PurePriorSignOPT3_20K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4}}}$ | {PurePriorSignOPT4_1K} | {PurePriorSignOPT4_3K} | {PurePriorSignOPT4_5K} | {PurePriorSignOPT4_7K} | {PurePriorSignOPT4_8K} | {PurePriorSignOPT4_10K} |  {PurePriorSignOPT4_12K} |  {PurePriorSignOPT4_15K} |  {PurePriorSignOPT4_18K} |  {PurePriorSignOPT4_20K} |
            | $\text{{Pure-Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4},{surrogate_arch5}}}$ | {PurePriorSignOPT5_1K} | {PurePriorSignOPT5_3K} | {PurePriorSignOPT5_5K} | {PurePriorSignOPT5_7K} | {PurePriorSignOPT5_8K} | {PurePriorSignOPT5_10K} |  {PurePriorSignOPT5_12K} |  {PurePriorSignOPT5_15K} |  {PurePriorSignOPT5_18K} |  {PurePriorSignOPT5_20K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1}}}$ | {PurePriorOPT1_1K} | {PurePriorOPT1_3K} | {PurePriorOPT1_5K} | {PurePriorOPT1_7K} | {PurePriorOPT1_8K} | {PurePriorOPT1_10K} |  {PurePriorOPT1_12K} |  {PurePriorOPT1_15K} |  {PurePriorOPT1_18K} |  {PurePriorOPT1_20K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2}}}$ | {PurePriorOPT2_1K} | {PurePriorOPT2_3K} | {PurePriorOPT2_5K} | {PurePriorOPT2_7K} | {PurePriorOPT2_8K} | {PurePriorOPT2_10K} |  {PurePriorOPT2_12K} |  {PurePriorOPT2_15K} |  {PurePriorOPT2_18K} |  {PurePriorOPT2_20K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3}}}$ | {PurePriorOPT3_1K} | {PurePriorOPT3_3K} | {PurePriorOPT3_5K} | {PurePriorOPT3_7K} | {PurePriorOPT3_8K} | {PurePriorOPT3_10K} |  {PurePriorOPT3_12K} |  {PurePriorOPT3_15K} |  {PurePriorOPT3_18K} |  {PurePriorOPT3_20K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4}}}$ | {PurePriorOPT4_1K} | {PurePriorOPT4_3K} | {PurePriorOPT4_5K} | {PurePriorOPT4_7K} | {PurePriorOPT4_8K} | {PurePriorOPT4_10K} |  {PurePriorOPT4_12K} |  {PurePriorOPT4_15K} |  {PurePriorOPT4_18K} |  {PurePriorOPT4_20K} |
            | $\text{{Pure-Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2},{surrogate_arch3},{surrogate_arch4},{surrogate_arch5}}}$ | {PurePriorOPT5_1K} | {PurePriorOPT5_3K} | {PurePriorOPT5_5K} | {PurePriorOPT5_7K} | {PurePriorOPT5_8K} | {PurePriorOPT5_10K} |  {PurePriorOPT5_12K} |  {PurePriorOPT5_15K} |  {PurePriorOPT5_18K} |  {PurePriorOPT5_20K} |
            | $\text{{Prior-Sign-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2}}}$ | {PriorSignOPT2_1K} | {PriorSignOPT2_3K} | {PriorSignOPT2_5K} | {PriorSignOPT2_7K} | {PriorSignOPT2_8K} | {PriorSignOPT2_10K} | {PriorSignOPT2_12K} | {PriorSignOPT2_15K} | {PriorSignOPT2_18K} | {PriorSignOPT2_20K} | 
            | $\text{{Prior-OPT}} _ \text{{{surrogate_arch1},{surrogate_arch2}}}$ | {PriorOPT2_1K} | {PriorOPT2_3K} | {PriorOPT2_5K} |  {PriorOPT2_7K} | {PriorOPT2_8K} | {PriorOPT2_10K} |  {PriorOPT2_12K} | {PriorOPT2_15K} | {PriorOPT2_18K} | {PriorOPT2_20K} | 
             """.format(
        surrogate_arch1=surrogate_arch1,
        surrogate_arch2=surrogate_arch2,
        surrogate_arch3=surrogate_arch3,
        surrogate_arch4=surrogate_arch4,
        surrogate_arch5=surrogate_arch5,

        SignOPT_1K=result["SignOPT"][1000],
        SignOPT_3K=result["SignOPT"][3000],
        SignOPT_5K=result["SignOPT"][5000],
        SignOPT_7K=result["SignOPT"][7000],
        SignOPT_8K=result["SignOPT"][8000],
        SignOPT_10K=result["SignOPT"][10000],
        SignOPT_12K=result["SignOPT"][12000],
        SignOPT_15K=result["SignOPT"][15000],
        SignOPT_18K=result["SignOPT"][18000],
        SignOPT_20K=result["SignOPT"][20000],

        PriorSignOPT2_1K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            1000],
        PriorSignOPT2_3K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            3000],
        PriorSignOPT2_5K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            5000],
        PriorSignOPT2_7K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            7000],
        PriorSignOPT2_8K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            8000],
        PriorSignOPT2_10K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            10000],
        PriorSignOPT2_12K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            12000],
        PriorSignOPT2_15K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            15000],
        PriorSignOPT2_18K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            18000],
        PriorSignOPT2_20K=result[
            "PriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            20000],

        PriorOPT2_1K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            1000],
        PriorOPT2_3K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            3000],
        PriorOPT2_5K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            5000],
        PriorOPT2_7K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            7000],
        PriorOPT2_8K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            8000],
        PriorOPT2_10K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
                10000],
        PriorOPT2_12K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            12000],
        PriorOPT2_15K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            15000],
        PriorOPT2_18K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            18000],
        PriorOPT2_20K=result[
            "PriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][
            20000],

        PurePriorSignOPT1_1K=result["PurePriorSignOPTwith" + surrogate_arch1][1000],
        PurePriorSignOPT1_3K=result["PurePriorSignOPTwith" + surrogate_arch1][3000],
        PurePriorSignOPT1_5K=result["PurePriorSignOPTwith" + surrogate_arch1][5000],
        PurePriorSignOPT1_7K=result["PurePriorSignOPTwith" + surrogate_arch1][7000],
        PurePriorSignOPT1_8K=result["PurePriorSignOPTwith" + surrogate_arch1][8000],
        PurePriorSignOPT1_10K=result["PurePriorSignOPTwith" + surrogate_arch1][10000],
        PurePriorSignOPT1_12K=result["PurePriorSignOPTwith" + surrogate_arch1][12000],
        PurePriorSignOPT1_15K=result["PurePriorSignOPTwith" + surrogate_arch1][15000],
        PurePriorSignOPT1_18K=result["PurePriorSignOPTwith" + surrogate_arch1][18000],
        PurePriorSignOPT1_20K=result["PurePriorSignOPTwith" + surrogate_arch1][20000],

        PurePriorSignOPT2_1K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][1000],
        PurePriorSignOPT2_3K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][3000],
        PurePriorSignOPT2_5K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][5000],
        PurePriorSignOPT2_7K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][7000],
        PurePriorSignOPT2_8K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][8000],
        PurePriorSignOPT2_10K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][10000],
        PurePriorSignOPT2_12K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][12000],
        PurePriorSignOPT2_15K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][15000],
        PurePriorSignOPT2_18K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][18000],
        PurePriorSignOPT2_20K=result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2][20000],

        PurePriorSignOPT3_1K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][1000],
        PurePriorSignOPT3_3K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][3000],
        PurePriorSignOPT3_5K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][5000],
        PurePriorSignOPT3_7K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][7000],
        PurePriorSignOPT3_8K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][8000],
        PurePriorSignOPT3_10K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][10000],
        PurePriorSignOPT3_12K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][12000],
        PurePriorSignOPT3_15K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][15000],
        PurePriorSignOPT3_18K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][18000],
        PurePriorSignOPT3_20K=
        result["PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][20000],

        PurePriorSignOPT4_1K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            1000],
        PurePriorSignOPT4_3K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            3000],
        PurePriorSignOPT4_5K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            5000],
        PurePriorSignOPT4_7K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            7000],
        PurePriorSignOPT4_8K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            8000],
        PurePriorSignOPT4_10K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            10000],
        PurePriorSignOPT4_12K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            12000],
        PurePriorSignOPT4_15K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            15000],
        PurePriorSignOPT4_18K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            18000],
        PurePriorSignOPT4_20K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            20000],

        PurePriorSignOPT5_1K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            1000],
        PurePriorSignOPT5_3K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            3000],
        PurePriorSignOPT5_5K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            5000],
        PurePriorSignOPT5_7K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            7000],
        PurePriorSignOPT5_8K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            8000],
        PurePriorSignOPT5_10K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            10000],
        PurePriorSignOPT5_12K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            12000],
        PurePriorSignOPT5_15K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            15000],
        PurePriorSignOPT5_18K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            18000],
        PurePriorSignOPT5_20K=result[
            "PurePriorSignOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            20000],

        PurePriorOPT1_1K=result["PurePriorOPTwith" + surrogate_arch1][1000],
        PurePriorOPT1_3K=result["PurePriorOPTwith" + surrogate_arch1][3000],
        PurePriorOPT1_5K=result["PurePriorOPTwith" + surrogate_arch1][5000],
        PurePriorOPT1_7K=result["PurePriorOPTwith" + surrogate_arch1][7000],
        PurePriorOPT1_8K=result["PurePriorOPTwith" + surrogate_arch1][8000],
        PurePriorOPT1_10K=result["PurePriorOPTwith" + surrogate_arch1][10000],
        PurePriorOPT1_12K=result["PurePriorOPTwith" + surrogate_arch1][12000],
        PurePriorOPT1_15K=result["PurePriorOPTwith" + surrogate_arch1][15000],
        PurePriorOPT1_18K=result["PurePriorOPTwith" + surrogate_arch1][18000],
        PurePriorOPT1_20K=result["PurePriorOPTwith" + surrogate_arch1][20000],

        PurePriorOPT2_1K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][1000],
        PurePriorOPT2_3K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][3000],
        PurePriorOPT2_5K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][5000],
        PurePriorOPT2_7K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][7000],
        PurePriorOPT2_8K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][8000],
        PurePriorOPT2_10K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][10000],
        PurePriorOPT2_12K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][12000],
        PurePriorOPT2_15K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][15000],
        PurePriorOPT2_18K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][18000],
        PurePriorOPT2_20K=result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2][20000],

        PurePriorOPT3_1K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][1000],
        PurePriorOPT3_3K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][3000],
        PurePriorOPT3_5K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][5000],
        PurePriorOPT3_7K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][7000],
        PurePriorOPT3_8K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][8000],
        PurePriorOPT3_10K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][10000],
        PurePriorOPT3_12K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][12000],
        PurePriorOPT3_15K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][15000],
        PurePriorOPT3_18K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][18000],
        PurePriorOPT3_20K=
        result["PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3][20000],

        PurePriorOPT4_1K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            1000],
        PurePriorOPT4_3K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            3000],
        PurePriorOPT4_5K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            5000],
        PurePriorOPT4_7K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            7000],
        PurePriorOPT4_8K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            8000],
        PurePriorOPT4_10K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            10000],
        PurePriorOPT4_12K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            12000],
        PurePriorOPT4_15K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            15000],
        PurePriorOPT4_18K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            18000],
        PurePriorOPT4_20K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4][
            20000],

        PurePriorOPT5_1K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            1000],
        PurePriorOPT5_3K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            3000],
        PurePriorOPT5_5K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            5000],
        PurePriorOPT5_7K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            7000],
        PurePriorOPT5_8K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            8000],
        PurePriorOPT5_10K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            10000],
        PurePriorOPT5_12K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            12000],
        PurePriorOPT5_15K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            15000],
        PurePriorOPT5_18K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            18000],
        PurePriorOPT5_20K=result[
            "PurePriorOPTwith" + surrogate_arch1 + "," + surrogate_arch2 + "," + surrogate_arch3 + "," + surrogate_arch4 + "," + surrogate_arch5][
            20000],
    ))


def draw_query_to_asr_table(result, surrogate_arch):
    print(r"""
            | Method | @1K | @3K | @5K | @7K |  @8K |  @10K |
            | :- | :- |:- | :- | :- | :- | :- |
            | HSJA | {HSJA_1K} |{HSJA_3K} | {HSJA_5K} | {HSJA_7K} | {HSJA_8K} | {HSJA_10K} |
            | TA | {TA_1K} |{TA_3K} | {TA_5K} | {TA_7K} | {TA_8K} | {TA_10K} |
            | G-TA | {GTA_1K} |{GTA_3K} | {GTA_5K} | {GTA_7K} | {GTA_8K} | {GTA_10K} |
            | GeoDA | {GeoDA_1K} |{GeoDA_3K} | {GeoDA_5K} | {GeoDA_7K} | {GeoDA_8K} | {GeoDA_10K} |
            | Evolutionary | {Evolutionary_1K} |{Evolutionary_3K} | {Evolutionary_5K} | {Evolutionary_7K} | {Evolutionary_8K} | {Evolutionary_10K} |
            | SurFree | {SurFree_1K} |{SurFree_3K} | {SurFree_5K} | {SurFree_7K} | {SurFree_8K} | {SurFree_10K} |
            | Triangle Attack | {TriA_1K} |{TriA_3K} | {TriA_5K} | {TriA_7K} | {TriA_8K} | {TriA_10K} |
            | SVM-OPT | {SVMOPT_1K} |{SVMOPT_3K} | {SVMOPT_5K} | {SVMOPT_7K} | {SVMOPT_8K} | {SVMOPT_10K} |
            | Sign-OPT  | {SignOPT_1K} |{SignOPT_3K} | {SignOPT_5K} | {SignOPT_7K} | {SignOPT_8K} | {SignOPT_10K} |
            | $\text{{SQBA}}_\text{{{surrogate_arch}}}$ | {SQBA_1K} |{SQBA_3K} | {SQBA_5K} | {SQBA_7K} | {SQBA_8K} | {SQBA_10K} |
            | $\text{{BBA}}_\text{{{surrogate_arch}}}$ | {BBA_1K} |{BBA_3K} | {BBA_5K} | {BBA_7K} | {BBA_8K} | {BBA_10K} |
            | CGBA-H | {CGBA_1K} | {CGBA_3K} | {CGBA_5K} | {CGBA_7K} | {CGBA_8K} | {CGBA_10K} |
            | AHA | {AHA_1K} | {AHA_3K} | {AHA_5K} | {AHA_7K} | {AHA_8K} | {AHA_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{{surrogate_arch}}}$ | {PriorSignOPT_1K} | {PriorSignOPT_3K} | {PriorSignOPT_5K} | {PriorSignOPT_7K} | {PriorSignOPT_8K} | {PriorSignOPT_10K} |
            | $\text{{Prior-OPT}} _ \text{{{surrogate_arch}}}$ | {PriorOPT_1K} | {PriorOPT_3K} | {PriorOPT_5K} |  {PriorOPT_7K} | {PriorOPT_8K} | {PriorOPT_10K} |
                                """.format(
        surrogate_arch=surrogate_arch,
        HSJA_1K=result["HSJA"][1000],
        HSJA_3K=result["HSJA"][3000],
        HSJA_5K=result["HSJA"][5000],
        HSJA_7K=result["HSJA"][7000],
        HSJA_8K=result["HSJA"][8000],
        HSJA_10K=result["HSJA"][10000],

        TA_1K=result["TA"][1000],
        TA_3K=result["TA"][3000],
        TA_5K=result["TA"][5000],
        TA_7K=result["TA"][7000],
        TA_8K=result["TA"][8000],
        TA_10K=result["TA"][10000],

        GTA_1K=result["G-TA"][1000],
        GTA_3K=result["G-TA"][3000],
        GTA_5K=result["G-TA"][5000],
        GTA_7K=result["G-TA"][7000],
        GTA_8K=result["G-TA"][8000],
        GTA_10K=result["G-TA"][10000],

        GeoDA_1K=result["GeoDA"][1000],
        GeoDA_3K=result["GeoDA"][3000],
        GeoDA_5K=result["GeoDA"][5000],
        GeoDA_7K=result["GeoDA"][7000],
        GeoDA_8K=result["GeoDA"][8000],
        GeoDA_10K=result["GeoDA"][10000],

        Evolutionary_1K=result["Evolutionary"][1000],
        Evolutionary_3K=result["Evolutionary"][3000],
        Evolutionary_5K=result["Evolutionary"][5000],
        Evolutionary_7K=result["Evolutionary"][7000],
        Evolutionary_8K=result["Evolutionary"][8000],
        Evolutionary_10K=result["Evolutionary"][10000],

        SurFree_1K=result["SurFree"][1000],
        SurFree_3K=result["SurFree"][3000],
        SurFree_5K=result["SurFree"][5000],
        SurFree_7K=result["SurFree"][7000],
        SurFree_8K=result["SurFree"][8000],
        SurFree_10K=result["SurFree"][10000],

        TriA_1K=result["Triangle Attack"][1000],
        TriA_3K=result["Triangle Attack"][3000],
        TriA_5K=result["Triangle Attack"][5000],
        TriA_7K=result["Triangle Attack"][7000],
        TriA_8K=result["Triangle Attack"][8000],
        TriA_10K=result["Triangle Attack"][10000],

        SQBA_1K=result["SQBAwith" + surrogate_arch][1000],
        SQBA_3K=result["SQBAwith" + surrogate_arch][3000],
        SQBA_5K=result["SQBAwith" + surrogate_arch][5000],
        SQBA_7K=result["SQBAwith" + surrogate_arch][7000],
        SQBA_8K=result["SQBAwith" + surrogate_arch][8000],
        SQBA_10K=result["SQBAwith" + surrogate_arch][10000],

        BBA_1K=result["BBAwith" + surrogate_arch][1000],
        BBA_3K=result["BBAwith" + surrogate_arch][3000],
        BBA_5K=result["BBAwith" + surrogate_arch][5000],
        BBA_7K=result["BBAwith" + surrogate_arch][7000],
        BBA_8K=result["BBAwith" + surrogate_arch][8000],
        BBA_10K=result["BBAwith" + surrogate_arch][10000],

        SignOPT_1K=result["SignOPT"][1000],
        SignOPT_3K=result["SignOPT"][3000],
        SignOPT_5K=result["SignOPT"][5000],
        SignOPT_7K=result["SignOPT"][7000],
        SignOPT_8K=result["SignOPT"][8000],
        SignOPT_10K=result["SignOPT"][10000],

        SVMOPT_1K=result["SVMOPT"][1000],
        SVMOPT_3K=result["SVMOPT"][3000],
        SVMOPT_5K=result["SVMOPT"][5000],
        SVMOPT_7K=result["SVMOPT"][7000],
        SVMOPT_8K=result["SVMOPT"][8000],
        SVMOPT_10K=result["SVMOPT"][10000],

        CGBA_1K=result["CGBA-H"][1000],
        CGBA_3K=result["CGBA-H"][3000],
        CGBA_5K=result["CGBA-H"][5000],
        CGBA_7K=result["CGBA-H"][7000],
        CGBA_8K=result["CGBA-H"][8000],
        CGBA_10K=result["CGBA-H"][10000],

        AHA_1K=result["AHA"][1000],
        AHA_3K=result["AHA"][3000],
        AHA_5K=result["AHA"][5000],
        AHA_7K=result["AHA"][7000],
        AHA_8K=result["AHA"][8000],
        AHA_10K=result["AHA"][10000],

        PriorSignOPT_1K=result["PriorSignOPTwith" + surrogate_arch][1000],
        PriorSignOPT_3K=result["PriorSignOPTwith" + surrogate_arch][3000],
        PriorSignOPT_5K=result["PriorSignOPTwith" + surrogate_arch][5000],
        PriorSignOPT_7K=result["PriorSignOPTwith" + surrogate_arch][7000],
        PriorSignOPT_8K=result["PriorSignOPTwith" + surrogate_arch][8000],
        PriorSignOPT_10K=result["PriorSignOPTwith" + surrogate_arch][10000],

        PriorOPT_1K=result["PriorOPTwith" + surrogate_arch][1000],
        PriorOPT_3K=result["PriorOPTwith" + surrogate_arch][3000],
        PriorOPT_5K=result["PriorOPTwith" + surrogate_arch][5000],
        PriorOPT_7K=result["PriorOPTwith" + surrogate_arch][7000],
        PriorOPT_8K=result["PriorOPTwith" + surrogate_arch][8000],
        PriorOPT_10K=result["PriorOPTwith" + surrogate_arch][10000],

        # PriorSignOPT_theta_1K=result["PriorSignOPTwithPGD" + surrogate_arch][1000],
        # PriorSignOPT_theta_3K=result["PriorSignOPTwithPGD" + surrogate_arch][3000],
        # PriorSignOPT_theta_5K=result["PriorSignOPTwithPGD" + surrogate_arch][5000],
        # PriorSignOPT_theta_7K=result["PriorSignOPTwithPGD" + surrogate_arch][7000],
        # PriorSignOPT_theta_8K=result["PriorSignOPTwithPGD" + surrogate_arch][8000],
        # PriorSignOPT_theta_10K=result["PriorSignOPTwithPGD" + surrogate_arch][10000],
        #
        # PriorOPT_theta_1K=result["PriorOPTwithPGD" + surrogate_arch][1000],
        # PriorOPT_theta_3K=result["PriorOPTwithPGD" + surrogate_arch][3000],
        # PriorOPT_theta_5K=result["PriorOPTwithPGD" + surrogate_arch][5000],
        # PriorOPT_theta_7K=result["PriorOPTwithPGD" + surrogate_arch][7000],
        # PriorOPT_theta_8K=result["PriorOPTwithPGD" + surrogate_arch][8000],
        # PriorOPT_theta_10K=result["PriorOPTwithPGD" + surrogate_arch][10000],
    )
    )


def draw_mean_queries_table(result):
    print(r"""
            | Method | Inception-V3 | Inception-V4 | ResNet-101 | ResNeXt-101(64x4d) | SENet-154 | ViT | GC ViT |  Swin Transformer | 
            | :- | :- |:- | :- | :- | :- | :- | :- | :- |
            | HSJA | {HSJA_inceptionv3} |{HSJA_inceptionv4} | {HSJA_resnet101} | {HSJA_resnext101_64x4d} | {HSJA_senet154} | {HSJA_jx_vit} | {HSJA_gcvit_base} | {HSJA_swin_base_patch4_window7_224} |
            | TA |  {TA_inceptionv3} |{TA_inceptionv4} | {TA_resnet101} | {TA_resnext101_64x4d} | {TA_senet154} | {TA_jx_vit} | {TA_gcvit_base} | {TA_swin_base_patch4_window7_224} |
            | G-TA |  {GTA_inceptionv3} |{GTA_inceptionv4} | {GTA_resnet101} | {GTA_resnext101_64x4d} | {GTA_senet154} | {GTA_jx_vit} | {GTA_gcvit_base} | {GTA_swin_base_patch4_window7_224} |
            | GeoDA | {GeoDA_inceptionv3} |{GeoDA_inceptionv4} | {GeoDA_resnet101} | {GeoDA_resnext101_64x4d} | {GeoDA_senet154} | {GeoDA_jx_vit} | {GeoDA_gcvit_base} | {GeoDA_swin_base_patch4_window7_224} |
            | Evolutionary | {Evolutionary_inceptionv3} |{Evolutionary_inceptionv4} | {Evolutionary_resnet101} | {Evolutionary_resnext101_64x4d} | {Evolutionary_senet154} | {Evolutionary_jx_vit} | {Evolutionary_gcvit_base} | {Evolutionary_swin_base_patch4_window7_224} |
            | SurFree | {SurFree_inceptionv3} |{SurFree_inceptionv4} | {SurFree_resnet101} | {SurFree_resnext101_64x4d} | {SurFree_senet154} | {SurFree_jx_vit} | {SurFree_gcvit_base} | {SurFree_swin_base_patch4_window7_224} |
            | Triangle Attack | {TriangleAttack_inceptionv3} |{TriangleAttack_inceptionv4} | {TriangleAttack_resnet101} | {TriangleAttack_resnext101_64x4d} | {TriangleAttack_senet154} | {TriangleAttack_jx_vit} | {TriangleAttack_gcvit_base} | {TriangleAttack_swin_base_patch4_window7_224} |
            | SVM-OPT | {SVMOPT_inceptionv3} |{SVMOPT_inceptionv4} | {SVMOPT_resnet101} | {SVMOPT_resnext101_64x4d} | {SVMOPT_senet154} | {SVMOPT_jx_vit} | {SVMOPT_gcvit_base} | {SVMOPT_swin_base_patch4_window7_224} |
            | Sign-OPT  | {SignOPT_inceptionv3} |{SignOPT_inceptionv4} | {SignOPT_resnet101} | {SignOPT_resnext101_64x4d} | {SignOPT_senet154} | {SignOPT_jx_vit} | {SignOPT_gcvit_base} | {SignOPT_swin_base_patch4_window7_224} |
            | SQBA  | {SQBA_inceptionv3} |{SQBA_inceptionv4} | {SQBA_resnet101} | {SQBA_resnext101_64x4d} | {SQBA_senet154} | {SQBA_jx_vit} | {SQBA_gcvit_base} | {SQBA_swin_base_patch4_window7_224} |
            | BBA | {BBA_inceptionv3} |{BBA_inceptionv4} | {BBA_resnet101} | {BBA_resnext101_64x4d} | {BBA_senet154} | {BBA_jx_vit} | {BBA_gcvit_base} | {BBA_swin_base_patch4_window7_224} |
            | Prior-Sign-OPT | {PriorSignOPTwithIncResV2_inceptionv3} |{PriorSignOPTwithIncResV2_inceptionv4} | {PriorSignOPTwithResNet50_resnet101} | {PriorSignOPTwithResNet50_resnext101_64x4d} | {PriorSignOPTwithResNet50_senet154} | {PriorSignOPTwithResNet50_jx_vit} | {PriorSignOPTwithResNet50_gcvit_base} | {PriorSignOPTwithResNet50_swin_base_patch4_window7_224} |
            | $\text{{Prior-Sign-OPT}} _ {{\theta _ 0^\text{{PGD}}}}$ | {PriorSignOPTwithPGDIncResV2_inceptionv3} |{PriorSignOPTwithPGDIncResV2_inceptionv4} | {PriorSignOPTwithPGDResNet50_resnet101} | {PriorSignOPTwithPGDResNet50_resnext101_64x4d} | {PriorSignOPTwithPGDResNet50_senet154} | {PriorSignOPTwithPGDResNet50_jx_vit} | {PriorSignOPTwithPGDResNet50_gcvit_base} | {PriorSignOPTwithPGDResNet50_swin_base_patch4_window7_224} |
            | Prior-OPT | {PriorOPTwithIncResV2_inceptionv3} |{PriorOPTwithIncResV2_inceptionv4} | {PriorOPTwithResNet50_resnet101} | {PriorOPTwithResNet50_resnext101_64x4d} | {PriorOPTwithResNet50_senet154} | {PriorOPTwithResNet50_jx_vit} | {PriorOPTwithResNet50_gcvit_base} | {PriorOPTwithResNet50_swin_base_patch4_window7_224} |
            | $\text{{Prior-OPT}} _ {{\theta _ 0^\text{{PGD}}}}$ | {PriorOPTwithPGDIncResV2_inceptionv3} |{PriorOPTwithPGDIncResV2_inceptionv4} | {PriorOPTwithPGDResNet50_resnet101} | {PriorOPTwithPGDResNet50_resnext101_64x4d} | {PriorOPTwithPGDResNet50_senet154} | {PriorOPTwithPGDResNet50_jx_vit} | {PriorOPTwithPGDResNet50_gcvit_base} | {PriorOPTwithPGDResNet50_swin_base_patch4_window7_224} |
        """.format(
        HSJA_inceptionv3=result["HSJA"]["inceptionv3"],
        HSJA_inceptionv4=result["HSJA"]["inceptionv4"],
        HSJA_resnet101=result["HSJA"]["resnet101"],
        HSJA_resnext101_64x4d=result["HSJA"]["resnext101_64x4d"],
        HSJA_senet154=result["HSJA"]["senet154"],
        HSJA_jx_vit=result["HSJA"]["jx_vit"],
        HSJA_gcvit_base=result["HSJA"]["gcvit_base"],
        HSJA_swin_base_patch4_window7_224=result["HSJA"]["swin_base_patch4_window7_224"],

        TA_inceptionv3=result["TA"]["inceptionv3"],
        TA_inceptionv4=result["TA"]["inceptionv4"],
        TA_resnet101=result["TA"]["resnet101"],
        TA_resnext101_64x4d=result["TA"]["resnext101_64x4d"],
        TA_senet154=result["TA"]["senet154"],
        TA_jx_vit=result["TA"]["jx_vit"],
        TA_gcvit_base=result["TA"]["gcvit_base"],
        TA_swin_base_patch4_window7_224=result["TA"]["swin_base_patch4_window7_224"],

        GTA_inceptionv3=result["G-TA"]["inceptionv3"],
        GTA_inceptionv4=result["G-TA"]["inceptionv4"],
        GTA_resnet101=result["G-TA"]["resnet101"],
        GTA_resnext101_64x4d=result["G-TA"]["resnext101_64x4d"],
        GTA_senet154=result["G-TA"]["senet154"],
        GTA_jx_vit=result["G-TA"]["jx_vit"],
        GTA_gcvit_base=result["G-TA"]["gcvit_base"],
        GTA_swin_base_patch4_window7_224=result["G-TA"]["swin_base_patch4_window7_224"],

        GeoDA_inceptionv3=result["GeoDA"]["inceptionv3"],
        GeoDA_inceptionv4=result["GeoDA"]["inceptionv4"],
        GeoDA_resnet101=result["GeoDA"]["resnet101"],
        GeoDA_resnext101_64x4d=result["GeoDA"]["resnext101_64x4d"],
        GeoDA_senet154=result["GeoDA"]["senet154"],
        GeoDA_jx_vit=result["GeoDA"]["jx_vit"],
        GeoDA_gcvit_base=result["GeoDA"]["gcvit_base"],
        GeoDA_swin_base_patch4_window7_224=result["GeoDA"]["swin_base_patch4_window7_224"],

        Evolutionary_inceptionv3=result["Evolutionary"]["inceptionv3"],
        Evolutionary_inceptionv4=result["Evolutionary"]["inceptionv4"],
        Evolutionary_resnet101=result["Evolutionary"]["resnet101"],
        Evolutionary_resnext101_64x4d=result["Evolutionary"]["resnext101_64x4d"],
        Evolutionary_senet154=result["Evolutionary"]["senet154"],
        Evolutionary_jx_vit=result["Evolutionary"]["jx_vit"],
        Evolutionary_gcvit_base=result["Evolutionary"]["gcvit_base"],
        Evolutionary_swin_base_patch4_window7_224=result["Evolutionary"]["swin_base_patch4_window7_224"],

        SurFree_inceptionv3=result["SurFree"]["inceptionv3"],
        SurFree_inceptionv4=result["SurFree"]["inceptionv4"],
        SurFree_resnet101=result["SurFree"]["resnet101"],
        SurFree_resnext101_64x4d=result["SurFree"]["resnext101_64x4d"],
        SurFree_senet154=result["SurFree"]["senet154"],
        SurFree_jx_vit=result["SurFree"]["jx_vit"],
        SurFree_gcvit_base=result["SurFree"]["gcvit_base"],
        SurFree_swin_base_patch4_window7_224=result["SurFree"]["swin_base_patch4_window7_224"],

        TriangleAttack_inceptionv3=result["Triangle Attack"]["inceptionv3"],
        TriangleAttack_inceptionv4=result["Triangle Attack"]["inceptionv4"],
        TriangleAttack_resnet101=result["Triangle Attack"]["resnet101"],
        TriangleAttack_resnext101_64x4d=result["Triangle Attack"]["resnext101_64x4d"],
        TriangleAttack_senet154=result["Triangle Attack"]["senet154"],
        TriangleAttack_jx_vit=result["Triangle Attack"]["jx_vit"],
        TriangleAttack_gcvit_base=result["Triangle Attack"]["gcvit_base"],
        TriangleAttack_swin_base_patch4_window7_224=result["Triangle Attack"]["swin_base_patch4_window7_224"],

        SVMOPT_inceptionv3=result["SVMOPT"]["inceptionv3"],
        SVMOPT_inceptionv4=result["SVMOPT"]["inceptionv4"],
        SVMOPT_resnet101=result["SVMOPT"]["resnet101"],
        SVMOPT_resnext101_64x4d=result["SVMOPT"]["resnext101_64x4d"],
        SVMOPT_senet154=result["SVMOPT"]["senet154"],
        SVMOPT_jx_vit=result["SVMOPT"]["jx_vit"],
        SVMOPT_gcvit_base=result["SVMOPT"]["gcvit_base"],
        SVMOPT_swin_base_patch4_window7_224=result["SVMOPT"]["swin_base_patch4_window7_224"],

        SignOPT_inceptionv3=result["SignOPT"]["inceptionv3"],
        SignOPT_inceptionv4=result["SignOPT"]["inceptionv4"],
        SignOPT_resnet101=result["SignOPT"]["resnet101"],
        SignOPT_resnext101_64x4d=result["SignOPT"]["resnext101_64x4d"],
        SignOPT_senet154=result["SignOPT"]["senet154"],
        SignOPT_jx_vit=result["SignOPT"]["jx_vit"],
        SignOPT_gcvit_base=result["SignOPT"]["gcvit_base"],
        SignOPT_swin_base_patch4_window7_224=result["SignOPT"]["swin_base_patch4_window7_224"],

        SQBA_inceptionv3=result["SQBA"]["inceptionv3"],
        SQBA_inceptionv4=result["SQBA"]["inceptionv4"],
        SQBA_resnet101=result["SQBA"]["resnet101"],
        SQBA_resnext101_64x4d=result["SQBA"]["resnext101_64x4d"],
        SQBA_senet154=result["SQBA"]["senet154"],
        SQBA_jx_vit=result["SQBA"]["jx_vit"],
        SQBA_gcvit_base=result["SQBA"]["gcvit_base"],
        SQBA_swin_base_patch4_window7_224=result["SQBA"]["swin_base_patch4_window7_224"],

        BBA_inceptionv3=result["BBA"]["inceptionv3"],
        BBA_inceptionv4=result["BBA"]["inceptionv4"],
        BBA_resnet101=result["BBA"]["resnet101"],
        BBA_resnext101_64x4d=result["BBA"]["resnext101_64x4d"],
        BBA_senet154=result["BBA"]["senet154"],
        BBA_jx_vit=result["BBA"]["jx_vit"],
        BBA_gcvit_base=result["BBA"]["gcvit_base"],
        BBA_swin_base_patch4_window7_224=result["BBA"]["swin_base_patch4_window7_224"],

        PriorSignOPTwithIncResV2_inceptionv3=result["PriorSignOPTwithIncResV2"]["inceptionv3"],
        PriorSignOPTwithIncResV2_inceptionv4=result["PriorSignOPTwithIncResV2"]["inceptionv4"],
        PriorSignOPTwithResNet50_resnet101=result["PriorSignOPTwithResNet50"]["resnet101"],
        PriorSignOPTwithResNet50_resnext101_64x4d=result["PriorSignOPTwithResNet50"]["resnext101_64x4d"],
        PriorSignOPTwithResNet50_senet154=result["PriorSignOPTwithResNet50"]["senet154"],
        PriorSignOPTwithResNet50_jx_vit=result["PriorSignOPTwithResNet50"]["jx_vit"],
        PriorSignOPTwithResNet50_gcvit_base=result["PriorSignOPTwithResNet50"]["gcvit_base"],
        PriorSignOPTwithResNet50_swin_base_patch4_window7_224=result["PriorSignOPTwithResNet50"]["swin_base_patch4_window7_224"],


        PriorSignOPTwithPGDIncResV2_inceptionv3=result["PriorSignOPTwithPGDIncResV2"]["inceptionv3"],
        PriorSignOPTwithPGDIncResV2_inceptionv4=result["PriorSignOPTwithPGDIncResV2"]["inceptionv4"],
        PriorSignOPTwithPGDResNet50_resnet101=result["PriorSignOPTwithPGDResNet50"]["resnet101"],
        PriorSignOPTwithPGDResNet50_resnext101_64x4d=result["PriorSignOPTwithPGDResNet50"]["resnext101_64x4d"],
        PriorSignOPTwithPGDResNet50_senet154=result["PriorSignOPTwithPGDResNet50"]["senet154"],
        PriorSignOPTwithPGDResNet50_jx_vit=result["PriorSignOPTwithPGDResNet50"]["jx_vit"],
        PriorSignOPTwithPGDResNet50_gcvit_base=result["PriorSignOPTwithPGDResNet50"]["gcvit_base"],
        PriorSignOPTwithPGDResNet50_swin_base_patch4_window7_224=result["PriorSignOPTwithPGDResNet50"]["swin_base_patch4_window7_224"],

        PriorOPTwithIncResV2_inceptionv3=result["PriorOPTwithIncResV2"]["inceptionv3"],
        PriorOPTwithIncResV2_inceptionv4=result["PriorOPTwithIncResV2"]["inceptionv4"],
        PriorOPTwithResNet50_resnet101=result["PriorOPTwithResNet50"]["resnet101"],
        PriorOPTwithResNet50_resnext101_64x4d=result["PriorOPTwithResNet50"]["resnext101_64x4d"],
        PriorOPTwithResNet50_senet154=result["PriorOPTwithResNet50"]["senet154"],
        PriorOPTwithResNet50_jx_vit=result["PriorOPTwithResNet50"]["jx_vit"],
        PriorOPTwithResNet50_gcvit_base=result["PriorOPTwithResNet50"]["gcvit_base"],
        PriorOPTwithResNet50_swin_base_patch4_window7_224=result["PriorOPTwithResNet50"]["swin_base_patch4_window7_224"],


        PriorOPTwithPGDIncResV2_inceptionv3=result["PriorOPTwithPGDIncResV2"]["inceptionv3"],
        PriorOPTwithPGDIncResV2_inceptionv4=result["PriorOPTwithPGDIncResV2"]["inceptionv4"],
        PriorOPTwithPGDResNet50_resnet101=result["PriorOPTwithPGDResNet50"]["resnet101"],
        PriorOPTwithPGDResNet50_resnext101_64x4d=result["PriorOPTwithPGDResNet50"]["resnext101_64x4d"],
        PriorOPTwithPGDResNet50_senet154=result["PriorOPTwithPGDResNet50"]["senet154"],
        PriorOPTwithPGDResNet50_jx_vit=result["PriorOPTwithPGDResNet50"]["jx_vit"],
        PriorOPTwithPGDResNet50_gcvit_base=result["PriorOPTwithPGDResNet50"]["gcvit_base"],
        PriorOPTwithPGDResNet50_swin_base_patch4_window7_224=result["PriorOPTwithPGDResNet50"]["swin_base_patch4_window7_224"],
    )
    )


def draw_table_for_pure_prioropt_with_more_priors_result(result):
    print(r"""
            | Method | @1K | @3K | @5K |  @8K |  @10K |
            | :- | :- |:- | :- | :- | :- |
            | HSJA | {HSJA_1K} | {HSJA_3K} | {HSJA_5K} | {HSJA_8K} | {HSJA_10K} |
            | Sign-OPT | {SignOPT_1K} | {SignOPT_3K} | {SignOPT_5K} | {SignOPT_8K} | {SignOPT_10K} |
            | SVM-OPT | {SVMOPT_1K} | {SVMOPT_3K} | {SVMOPT_5K} | {SVMOPT_8K} | {SVMOPT_10K} |
            | $\text{{Pure-Prior-Sign-OPT}}_\text{{ResNet50}}$ | {PurePriorSignOPT1_1K} | {PurePriorSignOPT1_3K} | {PurePriorSignOPT1_5K} | {PurePriorSignOPT1_8K} | {PurePriorSignOPT1_10K} |
            | $\text{{Pure-Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154}}$  | {PurePriorSignOPT2_1K} | {PurePriorSignOPT2_3K} | {PurePriorSignOPT2_5K} | {PurePriorSignOPT2_8K} | {PurePriorSignOPT2_10K} |
            | $\text{{Pure-Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101}}$  | {PurePriorSignOPT3_1K} | {PurePriorSignOPT3_3K} | {PurePriorSignOPT3_5K} | {PurePriorSignOPT3_8K} | {PurePriorSignOPT3_10K} |
            | $\text{{Pure-Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13}}$   | {PurePriorSignOPT4_1K} | {PurePriorSignOPT4_3K} | {PurePriorSignOPT4_5K} | {PurePriorSignOPT4_8K} | {PurePriorSignOPT4_10K} |
            | $\text{{Pure-Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13\\&SqueezeNet_v1.1}}$   | {PurePriorSignOPT5_1K} | {PurePriorSignOPT5_3K} | {PurePriorSignOPT5_5K} | {PurePriorSignOPT5_8K} | {PurePriorSignOPT5_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50}}$ | {PriorSignOPT1_1K} | {PriorSignOPT1_3K} | {PriorSignOPT1_5K} | {PriorSignOPT1_8K} | {PriorSignOPT1_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154}}$  | {PriorSignOPT2_1K} | {PriorSignOPT2_3K} | {PriorSignOPT2_5K} | {PriorSignOPT2_8K} | {PriorSignOPT2_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101}}$  | {PriorSignOPT3_1K} | {PriorSignOPT3_3K} | {PriorSignOPT3_5K} | {PriorSignOPT3_8K} | {PriorSignOPT3_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13}}$   | {PriorSignOPT4_1K} | {PriorSignOPT4_3K} | {PriorSignOPT4_5K} | {PriorSignOPT4_8K} | {PriorSignOPT4_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13\\&SqueezeNet_v1.1}}$   | {PriorSignOPT5_1K} | {PriorSignOPT5_3K} | {PriorSignOPT5_5K} | {PriorSignOPT5_8K} | {PriorSignOPT5_10K} |
            | $\text{{Pure-Prior-OPT}}_\text{{ResNet50}}$ | {PurePriorOPT1_1K} | {PurePriorOPT1_3K} | {PurePriorOPT1_5K} | {PurePriorOPT1_8K} | {PurePriorOPT1_10K} |
            | $\text{{Pure-Prior-OPT}}_\text{{ResNet50\\&SENet154}}$  | {PurePriorOPT2_1K} | {PurePriorOPT2_3K} | {PurePriorOPT2_5K} | {PurePriorOPT2_8K} | {PurePriorOPT2_10K} |
            | $\text{{Pure-Prior-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101}}$  | {PurePriorOPT3_1K} | {PurePriorOPT3_3K} | {PurePriorOPT3_5K} | {PurePriorOPT3_8K} | {PurePriorOPT3_10K} |
            | $\text{{Pure-Prior-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13}}$   | {PurePriorOPT4_1K} | {PurePriorOPT4_3K} | {PurePriorOPT4_5K} | {PurePriorOPT4_8K} | {PurePriorOPT4_10K} |
            | $\text{{Pure-Prior-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13\\&SqueezeNet_v1.1}}$   | {PurePriorOPT5_1K} | {PurePriorOPT5_3K} | {PurePriorOPT5_5K} | {PurePriorOPT5_8K} | {PurePriorOPT5_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50}}$ | {PriorOPT1_1K} | {PriorOPT1_3K} | {PriorOPT1_5K} | {PriorOPT1_8K} | {PriorOPT1_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50\\&SENet154}}$  | {PriorOPT2_1K} | {PriorOPT2_3K} | {PriorOPT2_5K} | {PriorOPT2_8K} | {PriorOPT2_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101}}$  | {PriorOPT3_1K} | {PriorOPT3_3K} | {PriorOPT3_5K} | {PriorOPT3_8K} | {PriorOPT3_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13}}$   | {PriorOPT4_1K} | {PriorOPT4_3K} | {PriorOPT4_5K} | {PriorOPT4_8K} | {PriorOPT4_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13\\&SqueezeNet_v1.1}}$   | {PriorOPT5_1K} | {PriorOPT5_3K} | {PriorOPT5_5K} | {PriorOPT5_8K} | {PriorOPT5_10K} |
                                """.format(
        HSJA_1K=result["HSJA"][1000],
        HSJA_3K=result["HSJA"][3000],
        HSJA_5K=result["HSJA"][5000],
        HSJA_8K=result["HSJA"][8000],
        HSJA_10K=result["HSJA"][10000],

        SignOPT_1K=result["Sign-OPT"][1000],
        SignOPT_3K=result["Sign-OPT"][3000],
        SignOPT_5K=result["Sign-OPT"][5000],
        SignOPT_8K=result["Sign-OPT"][8000],
        SignOPT_10K=result["Sign-OPT"][10000],

        SVMOPT_1K=result["SVM-OPT"][1000],
        SVMOPT_3K=result["SVM-OPT"][3000],
        SVMOPT_5K=result["SVM-OPT"][5000],
        SVMOPT_8K=result["SVM-OPT"][8000],
        SVMOPT_10K=result["SVM-OPT"][10000],

        PurePriorSignOPT1_1K=result["Pure-Prior-Sign-OPT1"][1000],
        PurePriorSignOPT1_3K=result["Pure-Prior-Sign-OPT1"][3000],
        PurePriorSignOPT1_5K=result["Pure-Prior-Sign-OPT1"][5000],
        PurePriorSignOPT1_8K=result["Pure-Prior-Sign-OPT1"][8000],
        PurePriorSignOPT1_10K=result["Pure-Prior-Sign-OPT1"][10000],

        PurePriorSignOPT2_1K=result["Pure-Prior-Sign-OPT2"][1000],
        PurePriorSignOPT2_3K=result["Pure-Prior-Sign-OPT2"][3000],
        PurePriorSignOPT2_5K=result["Pure-Prior-Sign-OPT2"][5000],
        PurePriorSignOPT2_8K=result["Pure-Prior-Sign-OPT2"][8000],
        PurePriorSignOPT2_10K=result["Pure-Prior-Sign-OPT2"][10000],

        PurePriorSignOPT3_1K=result["Pure-Prior-Sign-OPT3"][1000],
        PurePriorSignOPT3_3K=result["Pure-Prior-Sign-OPT3"][3000],
        PurePriorSignOPT3_5K=result["Pure-Prior-Sign-OPT3"][5000],
        PurePriorSignOPT3_8K=result["Pure-Prior-Sign-OPT3"][8000],
        PurePriorSignOPT3_10K=result["Pure-Prior-Sign-OPT3"][10000],

        PurePriorSignOPT4_1K=result["Pure-Prior-Sign-OPT4"][1000],
        PurePriorSignOPT4_3K=result["Pure-Prior-Sign-OPT4"][3000],
        PurePriorSignOPT4_5K=result["Pure-Prior-Sign-OPT4"][5000],
        PurePriorSignOPT4_8K=result["Pure-Prior-Sign-OPT4"][8000],
        PurePriorSignOPT4_10K=result["Pure-Prior-Sign-OPT4"][10000],

        PurePriorSignOPT5_1K=result["Pure-Prior-Sign-OPT5"][1000],
        PurePriorSignOPT5_3K=result["Pure-Prior-Sign-OPT5"][3000],
        PurePriorSignOPT5_5K=result["Pure-Prior-Sign-OPT5"][5000],
        PurePriorSignOPT5_8K=result["Pure-Prior-Sign-OPT5"][8000],
        PurePriorSignOPT5_10K=result["Pure-Prior-Sign-OPT5"][10000],

        PriorSignOPT1_1K=result["Prior-Sign-OPT1"][1000],
        PriorSignOPT1_3K=result["Prior-Sign-OPT1"][3000],
        PriorSignOPT1_5K=result["Prior-Sign-OPT1"][5000],
        PriorSignOPT1_8K=result["Prior-Sign-OPT1"][8000],
        PriorSignOPT1_10K=result["Prior-Sign-OPT1"][10000],

        PriorSignOPT2_1K=result["Prior-Sign-OPT2"][1000],
        PriorSignOPT2_3K=result["Prior-Sign-OPT2"][3000],
        PriorSignOPT2_5K=result["Prior-Sign-OPT2"][5000],
        PriorSignOPT2_8K=result["Prior-Sign-OPT2"][8000],
        PriorSignOPT2_10K=result["Prior-Sign-OPT2"][10000],

        PriorSignOPT3_1K=result["Prior-Sign-OPT3"][1000],
        PriorSignOPT3_3K=result["Prior-Sign-OPT3"][3000],
        PriorSignOPT3_5K=result["Prior-Sign-OPT3"][5000],
        PriorSignOPT3_8K=result["Prior-Sign-OPT3"][8000],
        PriorSignOPT3_10K=result["Prior-Sign-OPT3"][10000],

        PriorSignOPT4_1K=result["Prior-Sign-OPT4"][1000],
        PriorSignOPT4_3K=result["Prior-Sign-OPT4"][3000],
        PriorSignOPT4_5K=result["Prior-Sign-OPT4"][5000],
        PriorSignOPT4_8K=result["Prior-Sign-OPT4"][8000],
        PriorSignOPT4_10K=result["Prior-Sign-OPT4"][10000],

        PriorSignOPT5_1K=result["Prior-Sign-OPT5"][1000],
        PriorSignOPT5_3K=result["Prior-Sign-OPT5"][3000],
        PriorSignOPT5_5K=result["Prior-Sign-OPT5"][5000],
        PriorSignOPT5_8K=result["Prior-Sign-OPT5"][8000],
        PriorSignOPT5_10K=result["Prior-Sign-OPT5"][10000],

        PurePriorOPT1_1K=result["Pure-Prior-OPT1"][1000],
        PurePriorOPT1_3K=result["Pure-Prior-OPT1"][3000],
        PurePriorOPT1_5K=result["Pure-Prior-OPT1"][5000],
        PurePriorOPT1_8K=result["Pure-Prior-OPT1"][8000],
        PurePriorOPT1_10K=result["Pure-Prior-OPT1"][10000],

        PurePriorOPT2_1K=result["Pure-Prior-OPT2"][1000],
        PurePriorOPT2_3K=result["Pure-Prior-OPT2"][3000],
        PurePriorOPT2_5K=result["Pure-Prior-OPT2"][5000],
        PurePriorOPT2_8K=result["Pure-Prior-OPT2"][8000],
        PurePriorOPT2_10K=result["Pure-Prior-OPT2"][10000],

        PurePriorOPT3_1K=result["Pure-Prior-OPT3"][1000],
        PurePriorOPT3_3K=result["Pure-Prior-OPT3"][3000],
        PurePriorOPT3_5K=result["Pure-Prior-OPT3"][5000],
        PurePriorOPT3_8K=result["Pure-Prior-OPT3"][8000],
        PurePriorOPT3_10K=result["Pure-Prior-OPT3"][10000],

        PurePriorOPT4_1K=result["Pure-Prior-OPT4"][1000],
        PurePriorOPT4_3K=result["Pure-Prior-OPT4"][3000],
        PurePriorOPT4_5K=result["Pure-Prior-OPT4"][5000],
        PurePriorOPT4_8K=result["Pure-Prior-OPT4"][8000],
        PurePriorOPT4_10K=result["Pure-Prior-OPT4"][10000],

        PurePriorOPT5_1K=result["Pure-Prior-OPT5"][1000],
        PurePriorOPT5_3K=result["Pure-Prior-OPT5"][3000],
        PurePriorOPT5_5K=result["Pure-Prior-OPT5"][5000],
        PurePriorOPT5_8K=result["Pure-Prior-OPT5"][8000],
        PurePriorOPT5_10K=result["Pure-Prior-OPT5"][10000],

        PriorOPT1_1K=result["Prior-OPT1"][1000],
        PriorOPT1_3K=result["Prior-OPT1"][3000],
        PriorOPT1_5K=result["Prior-OPT1"][5000],
        PriorOPT1_8K=result["Prior-OPT1"][8000],
        PriorOPT1_10K=result["Prior-OPT1"][10000],

        PriorOPT2_1K=result["Prior-OPT2"][1000],
        PriorOPT2_3K=result["Prior-OPT2"][3000],
        PriorOPT2_5K=result["Prior-OPT2"][5000],
        PriorOPT2_8K=result["Prior-OPT2"][8000],
        PriorOPT2_10K=result["Prior-OPT2"][10000],

        PriorOPT3_1K=result["Prior-OPT3"][1000],
        PriorOPT3_3K=result["Prior-OPT3"][3000],
        PriorOPT3_5K=result["Prior-OPT3"][5000],
        PriorOPT3_8K=result["Prior-OPT3"][8000],
        PriorOPT3_10K=result["Prior-OPT3"][10000],

        PriorOPT4_1K=result["Prior-OPT4"][1000],
        PriorOPT4_3K=result["Prior-OPT4"][3000],
        PriorOPT4_5K=result["Prior-OPT4"][5000],
        PriorOPT4_8K=result["Prior-OPT4"][8000],
        PriorOPT4_10K=result["Prior-OPT4"][10000],

        PriorOPT5_1K=result["Prior-OPT5"][1000],
        PriorOPT5_3K=result["Prior-OPT5"][3000],
        PriorOPT5_5K=result["Prior-OPT5"][5000],
        PriorOPT5_8K=result["Prior-OPT5"][8000],
        PriorOPT5_10K=result["Prior-OPT5"][10000],
    )
    )


def draw_table_for_more_priors_ASR_mean_queries(result):
    print(r"""
    | Metric | no prior(Sign-OPT) | 1 prior | 2 priors | 3 priors | 4 priors | 5 priors |
    | :- | :- |:- | :- | :- |:- | :- | 
    | Prior-Sign-OPT's ASR | {SignOPT_ASR} | {PriorSignOPT1_ASR} | {PriorSignOPT2_ASR} | {PriorSignOPT3_ASR} | {PriorSignOPT4_ASR} | {PriorSignOPT5_ASR} |
    | Prior-OPT's ASR | {SignOPT_ASR} | {PriorOPT1_ASR} | {PriorOPT2_ASR} | {PriorOPT3_ASR} | {PriorOPT4_ASR} | {PriorOPT5_ASR} |
    | Prior-Sign-OPT's Avg Queries | {SignOPT_queries} | {PriorSignOPT1_queries} | {PriorSignOPT2_queries} | {PriorSignOPT3_queries} | {PriorSignOPT4_queries} | {PriorSignOPT5_queries} |
    | Prior-OPT's Avg Queries | {SignOPT_queries} | {PriorOPT1_queries} | {PriorOPT2_queries} | {PriorOPT3_queries} | {PriorOPT4_queries} | {PriorOPT5_queries} |
    | Prior-Sign-OPT's Distortion | {SignOPT_distortion} | {PriorSignOPT1_distortion} | {PriorSignOPT2_distortion} | {PriorSignOPT3_distortion} | {PriorSignOPT4_distortion} | {PriorSignOPT5_distortion} |
    | Prior-OPT's Distortion | {SignOPT_distortion} | {PriorOPT1_distortion} | {PriorOPT2_distortion} | {PriorOPT3_distortion} | {PriorOPT4_distortion} | {PriorOPT5_distortion} |
     """.format(
    SignOPT_ASR=result["SignOPT"]["ASR"],
    SignOPT_queries=result["SignOPT"]["queries"],
    SignOPT_distortion = result["SignOPT"]["distortion"],

    PriorSignOPT1_ASR = result["PriorSignOPT1"]["ASR"],
    PriorSignOPT2_ASR = result["PriorSignOPT2"]["ASR"],
    PriorSignOPT3_ASR = result["PriorSignOPT3"]["ASR"],
    PriorSignOPT4_ASR = result["PriorSignOPT4"]["ASR"],
    PriorSignOPT5_ASR = result["PriorSignOPT5"]["ASR"],

    PriorOPT1_ASR = result["PriorOPT1"]["ASR"],
    PriorOPT2_ASR = result["PriorOPT2"]["ASR"],
    PriorOPT3_ASR = result["PriorOPT3"]["ASR"],
    PriorOPT4_ASR = result["PriorOPT4"]["ASR"],
    PriorOPT5_ASR = result["PriorOPT5"]["ASR"],

    PriorSignOPT1_queries = result["PriorSignOPT1"]["queries"],
    PriorSignOPT2_queries = result["PriorSignOPT2"]["queries"],
    PriorSignOPT3_queries = result["PriorSignOPT3"]["queries"],
    PriorSignOPT4_queries = result["PriorSignOPT4"]["queries"],
    PriorSignOPT5_queries = result["PriorSignOPT5"]["queries"],

    PriorOPT1_queries = result["PriorOPT1"]["queries"],
    PriorOPT2_queries = result["PriorOPT2"]["queries"],
    PriorOPT3_queries = result["PriorOPT3"]["queries"],
    PriorOPT4_queries = result["PriorOPT4"]["queries"],
    PriorOPT5_queries = result["PriorOPT5"]["queries"],

    PriorSignOPT1_distortion = result["PriorSignOPT1"]["distortion"],
    PriorSignOPT2_distortion = result["PriorSignOPT2"]["distortion"],
    PriorSignOPT3_distortion = result["PriorSignOPT3"]["distortion"],
    PriorSignOPT4_distortion = result["PriorSignOPT4"]["distortion"],
    PriorSignOPT5_distortion = result["PriorSignOPT5"]["distortion"],

    PriorOPT1_distortion = result["PriorOPT1"]["distortion"],
    PriorOPT2_distortion = result["PriorOPT2"]["distortion"],
    PriorOPT3_distortion = result["PriorOPT3"]["distortion"],
    PriorOPT4_distortion = result["PriorOPT4"]["distortion"],
    PriorOPT5_distortion = result["PriorOPT5"]["distortion"],
    )
    )

def get_success_distortion_threshold(dataset, arch):
    if dataset == "CIFAR-10":
        success_distortion_threshold = 1.0
        dim = 32 * 32 * 3
    else:
        if "inception" in arch or "xception" in arch or "ception" in arch:
            dim = 3 * 299 * 299
            print("{} is 299!".format(arch))
        else:
            dim = 3 * 224 * 224
        success_distortion_threshold = math.sqrt(0.001 * dim)
    print("arch:{}, dim:{}, Success distortion threshold:{}".format(arch, dim, success_distortion_threshold))
    return success_distortion_threshold

if __name__ == "__main__":
    dataset = "ImageNet"
    norm = "l2"
    targeted = False
    defense_model = False
    if "CIFAR" in dataset:
        archs = ['pyramidnet272']
    else:
        # archs = ["inceptionv3","inceptionv4","resnet101","resnext101_64x4d","senet154","jx_vit","gcvit_base","swin_base_patch4_window7_224"]
        archs = ["clip"]
    query_budgets = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    mean_query_all_results = defaultdict(dict)
    attack_success_rate_all_results = defaultdict(dict)
    # for arch in archs:
    #     print("=============={}=================".format(arch))
    #     success_distortion_threshold = get_success_distortion_threshold(dataset, arch)
    #     mean_query_result, query_to_asr_result, attack_success_rate_result, arch_to_surrogate_arch = fetch_all_json_content_given_contraint(dataset, norm, arch, False,
    #                                                         success_distortion_threshold,query_budgets, max(query_budgets))
    #     # draw_query_to_asr_table(query_to_asr_result[arch],arch_to_surrogate_arch[arch])
    #     mean_query_all_results[arch].update(mean_query_result[arch])
    #     attack_success_rate_all_results[arch].update(attack_success_rate_result[arch])

    # for arch in archs:
    #     print("=============={}=================".format(arch))
    #     success_distortion_threshold = get_success_distortion_threshold(dataset, arch)
    #     attack_success_rate_result = fetch_all_json_content_for_BASES(dataset, norm, arch, targeted,
    #                                                         success_distortion_threshold,query_budgets, max(query_budgets))
    #     # draw_query_to_asr_table(query_to_asr_result[arch],arch_to_surrogate_arch[arch])
    #     attack_success_rate_all_results[arch].update(attack_success_rate_result[arch])
    #
    # reversed_dict = defaultdict(dict)
    # for arch, methods in attack_success_rate_all_results.items():
    #     for method, value in methods.items():
    #         reversed_dict[method][arch] = value

    for arch in archs:
        # threshold = get_success_distortion_threshold(dataset, arch)
        max_queries = 10000 if not targeted else 20000
        print("=============={}=================".format(arch))
        threshold = get_success_distortion_threshold(dataset,arch)
        all_result = fetch_all_json_content_for_comparing_priors_ASR_mean_queries(dataset,norm, arch, targeted, threshold, max_queries)
        draw_table_for_more_priors_ASR_mean_queries(all_result)


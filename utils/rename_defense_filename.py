import os
import sys
import re
import json


def log_file_new_name(orig_file_path):
    assert orig_file_path.endswith('.log')
    orig_file_name = os.path.basename(orig_file_path)
    json_file_pattern = r".*?result will be saved to logs/(.*?)/(.*json)"
    find_json_file_name = None
    with open(orig_file_path, 'r') as file:
        for line in file.readlines():
            searcher = re.search(json_file_pattern, line.strip())
            if searcher:
                find_json_file_name = searcher.group(2)
                break
    assert find_json_file_name
    json_file_path = os.path.dirname(orig_file_path) + "/" + find_json_file_name
    if not os.path.exists(json_file_path):
        return None
    new_json_file_name = json_file_new_name(json_file_path)
    new_log_file_name = "run_" + os.path.splitext(new_json_file_name)[0] + ".log"
    # print("{} --> {}".format(orig_file_name, new_log_file_name))
    return new_log_file_name

def json_file_new_name(orig_file_path):
    # old filename: resnet50_adv_train_on_ImageNet_linf_8_div_255_result.json,       resnet-50_surrogates_resnet-110,densenet-bc-100-12_adv_train_result.json
    # new filename: resnet50(adv_train_linf_8_div_255)_surrogates_resnet-110.json , resnet50(adv_train_linf_4_div_255)_surrogates_resnet50,senet154.json,
    # resnet-50_adv_train_result.json -> resnet-50(adv_train_linf_8_div_255)_surrogates_resnet-110.json 或逗号隔开的多代理模型
    assert orig_file_path.endswith('.json')
    with open(orig_file_path, 'r') as file:
        data = json.load(file)
        arch = data['args']['arch']
        assert data["args"]["attack_defense"] == True
        if "surrogate_arch" in data["args"] and data["args"]["surrogate_arch"]:
            surrogate_archs = [data["args"]["surrogate_arch"]]
        elif "surrogate_archs" in data["args"]:
            surrogate_archs = data["args"]["surrogate_archs"]

        surrogate_str = []
        for idx, surrogate_arch in enumerate(surrogate_archs):
            if "surrogate_defense_models" in data["args"] and data['args']['surrogate_defense_models']:
                surrogate_defense_model = data['args']['surrogate_defense_models'][idx]
                if surrogate_defense_model.startswith("adv_train"):
                    surrogate_defense_model = "AT"
                if surrogate_defense_model == "AT" or surrogate_defense_model == "TRADES" or surrogate_defense_model == "feature_scatter":
                    if "surrogate_defense_norms" not in data['args']:
                        surrogate_defense_norm = "linf"
                    else:
                        surrogate_defense_norm = data['args']['surrogate_defense_norms'][idx]
                    if "surrogate_defense_eps" not in data['args']:
                        surrogate_defense_eps = "8_div_255"
                        if surrogate_defense_model == "feature_scatter":
                            surrogate_defense_eps = "16_div_255"
                    else:
                        surrogate_defense_eps = data['args']['surrogate_defense_eps'][idx]

                    surrogate_str.append("{surrogate_arch}({surrogate_defense_model}_{surrogate_defense_norm}_{surrogate_defense_eps})"
                                 .format(surrogate_arch=surrogate_arch, surrogate_defense_model=surrogate_defense_model,
                                         surrogate_defense_norm=surrogate_defense_norm,
                                         surrogate_defense_eps=surrogate_defense_eps))
                else:
                    surrogate_str.append(
                        "{surrogate_arch}({surrogate_defense_model})"
                        .format(surrogate_arch=surrogate_arch, surrogate_defense_model=surrogate_defense_model))

            else:
                surrogate_str.append(
                    "{surrogate_arch}".format(surrogate_arch=surrogate_arch))

        arch_defense_model = data["args"]["defense_model"]
        if arch_defense_model.startswith("adv_train"):
            arch_defense_model = "AT"
        if arch_defense_model == "AT" or arch_defense_model == "TRADES" or arch_defense_model == "feature_scatter":
            arch_defense_norm = data["args"]["defense_norm"]
            if arch_defense_norm == "linf" and not data['args']['defense_eps']:
                arch_defense_eps = "8_div_255"
                if arch_defense_model == "feature_scatter":
                    arch_defense_eps = "16_div_255"
            else:
                arch_defense_eps = data["args"]["defense_eps"]
            assert arch_defense_eps
            orig_filename = os.path.basename(orig_file_path)
            new_filename = "{arch}({arch_defense_model}_{arch_defense_norm}_{arch_defense_eps})_surrogates_{surrogate_archs}.json".format(
                arch=arch, arch_defense_model=arch_defense_model, arch_defense_norm=arch_defense_norm,
                arch_defense_eps=arch_defense_eps, surrogate_archs=",".join(surrogate_str))
        else:
            new_filename = "{arch}({arch_defense_model})_surrogates_{surrogate_archs}.json".format(
                arch=arch, arch_defense_model=arch_defense_model, surrogate_archs=",".join(surrogate_str))
    #print("{} --> {}".format(orig_filename, new_filename))
    return new_filename


if __name__ == '__main__':
    for dir_name in os.listdir("H:\\logs\\hard_label_attacks\\"):
        if dir_name.startswith("PriorSignOPT_on_defensive_model") or dir_name.startswith("PriorOPT_on_defensive_model"):
            if "CIFAR-10" not in dir_name:
                continue
            dir_name = "H:\\logs\\hard_label_attacks\\"+dir_name
            rename_list = []
            for filename in os.listdir(dir_name):
                if filename.endswith(".log"):
                    new_log_file_name = log_file_new_name(dir_name + "\\" + filename)
                    if new_log_file_name:
                        rename_list.append((filename, new_log_file_name))
                elif filename.endswith(".json"):
                    new_file_name = new_json_file_name = json_file_new_name(dir_name + "\\" + filename)
                    rename_list.append((filename, new_file_name))
            for old_filename, new_file_name in rename_list:
                new_file_path = dir_name + "\\" + new_file_name
                print("Renaming {} to {}".format(old_filename, new_file_name))
                os.rename(dir_name + "\\" + old_filename, new_file_path)



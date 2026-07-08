from .prior_gw import ProbDistContainer


def compare_model_parameters(parameter_dict: dict[str, list[str]]) -> dict[str, set[int]]:

    missing_params: dict[str, list] = dict()
    missing_params_inds: dict[str, list] = {model: [] for model in parameter_dict}
    parameter_sets = {model: set(params) for model, params in parameter_dict.items()}

    for model, params in parameter_sets.items():
        missing = set()
        
        for other_model, other_params in parameter_sets.items():

            if len(params) <= len(other_params):
                assert params.issubset(other_params), (
                    f"Model '{model}' has smaller or equal dimension than '{other_model}', "
                    f"but contains parameters not found in '{other_model}': {params - other_params} "
                )
                missing.update(other_params - params)
        
        missing = list(missing)
        missing_params[model]  = missing

        for p in missing:
            indices = [plist.index(p) for plist in parameter_dict.values() if p in plist]

            if len(set(indices))>1:
                raise IndexError(
                    f"Inconsistent position for parameter '{p}' across models: {indices}"
                    )       
            missing_params_inds[model].append(indices[0])
        
      
    return missing_params, missing_params_inds

def match_model_parameters(parameter_dict: dict[str, list[str]]) -> dict[str, list[str]]:

    parameter_dict = parameter_dict.copy()

    missing_params, missing_params_inds = compare_model_parameters(parameter_dict)
    
    for model in parameter_dict:
        for ind, p in zip(missing_params[model], missing_params_inds[model]):
            parameter_dict[model].insert(p, ind)
    

    if len({tuple(v) for v in parameter_dict.values()}) !=1:
        raise ValueError(f"Different parameters were provided across different models: {set(parameter_dict.values())}")
    
    return parameter_dict


def check_parameter_order(priors: dict[str, ProbDistContainer]) -> tuple:

    parameter_names = {model: list(prior.priors_in.keys()) for model, prior in priors.items()}

    parameter_combinations = {tuple(combo) for combo in parameter_names.values()}

    if len(parameter_combinations) > 1:
        raise ValueError(f"Parameter combinations are not identical across models: {parameter_combinations}")

    return priors, list(*parameter_combinations)
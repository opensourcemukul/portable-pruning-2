# import torch
# from ptflops import get_model_complexity_info

# def compute_flops(model, input_res=(3, 224, 224)):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     with torch.cuda.device(0) if device.type == "cuda" else torch.no_grad():
#         macs, params = get_model_complexity_info(
#             model, input_res,
#             as_strings=True,
#             print_per_layer_stat=False
#         )
#     return macs, params
import torch
from ptflops import get_model_complexity_info

def compute_flops(model, input_res=(3, 224, 224)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the proper context: either CUDA device if available or no_grad for CPU.
    with torch.cuda.device(0) if device.type == "cuda" else torch.no_grad():
        # Get raw numeric values rather than strings
        macs, params = get_model_complexity_info(
            model, input_res,
            as_strings=False,              # raw numbers returned
            print_per_layer_stat=False
        )
    # Convert MACs to GMac (i.e. billions of MACs) and parameters to Millions (M)
    macs_in_gmac = float(macs) / 1e9
    params_in_m = float(params) / 1e6
    return macs_in_gmac, params_in_m
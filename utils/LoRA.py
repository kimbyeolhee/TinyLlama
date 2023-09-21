import bitsandbytes as bnb


# code from https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    """
    Returns a list of all linear layer names in the model
    """
    cls = bnb.nn.Linear4bit # 4bit linear layer
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head") # ?
    return lora_module_names

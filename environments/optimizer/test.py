from modal.functions import Function

send_code = Function.from_name("optimizer-test", "send_code")

res = send_code.remote("""
import torch
import math

class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
""")

print(res)
from modal.functions import Function
from evaluator import OptimizerEvaluator

def score_optimizer(optimizer_code: str, architecture: str):
    """
    Test an optimizer implementation by sending it to the Modal sandbox environment.
    
    Args:
        optimizer_code (str): The optimizer code to test
        
    Returns:
        dict: Results containing stdout, stderr, code and filename
    """
    send_code = Function.from_name("optimizer-test", "send_code")

    response_obj = send_code.remote(optimizer_code)

    print(response_obj)

    evaluator = OptimizerEvaluator()

    validity = evaluator.check_validity(optimizer_code=optimizer_code, stdout=response_obj["stdout"], stderr=response_obj["stderr"])
    
    if not validity:
        return 0
    else:
        score = evaluator.score_optimizer(optimizer_code=optimizer_code, architecture=architecture)
        return score

if __name__ == "__main__":
    test_code = """
import torch
import math

class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
"""
    result = score_optimizer(test_code, "mnist")
    print(result)

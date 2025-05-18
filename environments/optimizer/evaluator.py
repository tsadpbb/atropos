from verdict.schema import Schema
from verdict import Pipeline, Layer
from verdict.common.judge import JudgeUnit
from verdict.scale import ContinuousScale
from verdict.transform import MaxPoolUnit


class OptimizerEvaluator:
    def __init__(self):
        self.pipeline = (
            Pipeline()
            >> Layer(
                JudgeUnit(scale=ContinuousScale(1, 10)).prompt(
                    (
                        "You are a judge that is an expert at evaluating optimizers for their novelty "
                        "as they will be accepted to a prestigious research conference. Given the following "
                        "optimizer code and its architecture/use-case, you must rate it on a scale of 1 to 10 "
                        "based on how novel it is and its impactfulness in speeding up model training. "
                        "Here is the code: {source.optimizer_code}\n"
                        "Here is the architecture: {source.architecture}"
                    )
                ),
                repeat=3,
            ).via("xai/grok-3-latest")
            >> MaxPoolUnit()
        )

    def run(self, optimizer_code: str, architecture: str) -> int:
        schema = Schema.of(
            optimizer_code=optimizer_code,
            architecture=architecture,
        )
        response, _ = self.pipeline.run(schema)
        final_score = self.__get_final_score(response)
        return final_score

    def __get_final_score(self, response: dict) -> float:
        return response.get("Pipeline_root.block.block.unit[Map MaxPool]_score", 0.0)


if __name__ == "__main__":
    evaluator = OptimizerEvaluator()

    optimizer_code = """
import torch

# Define parameter (requires_grad=True)
x = torch.tensor([0.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

for step in range(20):
    optimizer.zero_grad()
    loss = (x - 3) ** 2
    loss.backward()
    optimizer.step()
    print(f"Step {step + 1}: x = {x.item():.4f}, loss = {loss.item():.4f}")

print(f"\nOptimal x: {x.item():.4f}")
    """

    score = evaluator.run(optimizer_code=optimizer_code, architecture="MLP")
    print(score)

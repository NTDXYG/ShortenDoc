from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_self_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_self_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)


if __name__ == '__main__':
    dataset = 'odex'
    method = ''
    model = ''
    entry_point('./results/raw/humaneval/CodeQwen.jsonl', k='1',
                problem_file='./dataset/HumanEval.jsonl')
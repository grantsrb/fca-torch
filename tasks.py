from typing import List, Tuple, Callable
import numpy as np
from torch.utils.data import Dataset
import torch

POSSIBLE_OPERATIONS = [lambda x, y: x & y, lambda x, y: x | y, lambda x, y: x ^ y]
OPERATION2STRING = {
    POSSIBLE_OPERATIONS[0]: "and",
    POSSIBLE_OPERATIONS[1]: "or",
    POSSIBLE_OPERATIONS[2]: "xor",
}
STRING2OPERATION = {
    "and": POSSIBLE_OPERATIONS[0],
    "or":  POSSIBLE_OPERATIONS[1],
    "xor": POSSIBLE_OPERATIONS[2],
}

def logical_operation(op: Callable[[int, int], int], a: int, b: int) -> int:
    return op(a, b)

def hierarchical_logical_task(
        pairs: List[Tuple[int, int]],
        operations: List[Callable[[int, int], int]]) -> int:
    """
    Given a list of pairs of integers and a list of operations, this function
    will perform the operations in a hierarchical manner. The operations will be
    performed in pairs, and the result will be used in the next operation. This
    will continue until there is only one pair left. The final result will be
    returned.
    """
    if not pairs or not operations:
        raise ValueError("Pairs and operations must not be empty")
    
    while len(pairs) > 1:
        new_pairs = []
        for i in range(0, len(pairs), 2):
            if i + 1 < len(pairs):
                a = logical_operation(
                    operations.pop(0),
                    pairs[i][0],
                    pairs[i][1]
                )
                b = logical_operation(
                    operations.pop(0),
                    pairs[i+1][0],
                    pairs[i+1][1]
                )
                new_pairs.append((a, b))
            else:
                new_pairs.append(pairs[i])
        pairs = new_pairs
    
    final_pair = pairs[0]
    return logical_operation(operations.pop(0), final_pair[0], final_pair[1])

def generate_binary_pairs(n: int) -> List[Tuple[int, int]]:
    """
    This function will generate n pairs of integers with values 0 or 1.
    """
    return [(np.random.randint(2), np.random.randint(2)) for _ in range(n)]

def generate_random_operations(n: int) -> List[Callable[[int, int], int]]:
    """
    This function will generate n random operations.
    """
    return [POSSIBLE_OPERATIONS[np.random.randint(len(POSSIBLE_OPERATIONS))] for _ in range(n)]

def get_total_operations(n: int) -> int:
    """
    This function will return the total number of operations needed to solve the task.
    """
    return np.sum([n // 2**i for i in range(n)])+int(n%2!=0)

def generate_logical_task(
        n_pairs: int, n_samples: int, operations: List[Callable[[int, int], int]]=None
    ):
    """
    This function will generate a logical task with n pairs of integers and n*log2(n) random operations.
    """
    total_operations = get_total_operations(n_pairs)
    if operations is None:
        operations = generate_random_operations(total_operations)
    pairs = [generate_binary_pairs(n_pairs) for _ in range(n_samples)]
    outputs = [hierarchical_logical_task(p, operations.copy()) for p in pairs]
    return pairs, operations, outputs

def convert_to_string_dataset(input_pairs, outputs):
    """
    This function will convert the input pairs and outputs to a string dataset.
    """
    input_strings = []
    output_strings = []
    for p,o in zip(input_pairs, outputs):
        input_strings.append(
            " ".join([f"{x[0]} {x[1]}" for x in p])
        )
        output_strings.append(f"{o}")
    return input_strings, output_strings

def get_hierarchical_logical_task_dataset(
        n_pairs: int,
        n_samples: int,
        operations=None,
    ):
    pairs, operations, outputs = generate_logical_task(
        n_pairs,
        n_samples,
        operations=operations
    )
    input_strings,output_strings = convert_to_string_dataset(
        pairs, outputs)
    return {
        "input_strings": input_strings,
        "output_strings": output_strings,
    }, operations

def get_dataset(config):
    n_pairs = config["n_pairs"]
    n_samples = config["n_samples"]
    operations = config.get("operations", None)
    if operations and type(operations[0]) == str:
        operations = [STRING2OPERATION[o] for o in operations]
    return get_hierarchical_logical_task_dataset(
        n_pairs=n_pairs,
        n_samples=n_samples,
        operations=operations
    )

class HierarchicalLogicalTask(Dataset):
    def __init__(self, config):
        self.batch_size = config.get("batch_size", 128)
        self.data, self.operations = get_dataset(config)
        self.operation_names = [
            OPERATION2STRING[o] for o in self.operations
        ]
        self.vocab = config.get("vocab", set())
        for k in ["input_strings", "output_strings"]:
            self.data[k] = [
                s.split(" ") for s in self.data[k]
            ]
            self.vocab.update([w for s in self.data[k] for w in s])
        self.word2idx = config.get(
            "word2idx", 
            {w: i for i, w in enumerate(self.vocab)}
        )
        self.idx2word = config.get(
            "idx2word",
            {i: w for i, w in enumerate(self.vocab)}
        )
        self.n_input_tokens = len(self.data["input_strings"][0])
        self.vocab_size = len(self.vocab)
        self.data["input_ids"] = torch.LongTensor([
            [self.word2idx[w] for w in s]\
                for s in self.data["input_strings"]
        ])
        self.data["output_ids"] = torch.LongTensor([
            [self.word2idx[w] for w in s]\
                for s in self.data["output_strings"]
        ])
    
    def __len__(self):
        return len(self.data["input_ids"])
    
    def n_batches(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return len(self) // batch_size

    def __getitem__(self, idx):
        return self.data["input_ids"][idx], self.data["output_ids"][idx]

    def get_batches(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        perm = torch.randperm(len(self)).long()
        for i in range(self.n_batches(batch_size)):
            idxs = perm[i*batch_size:(i+1)*batch_size]
            yield {
                "input_ids": self.data["input_ids"][idxs],
                "output_ids": self.data["output_ids"][idxs],
            }


# Example usage:
if __name__ == "__main__":
    pairs = [(1, 0), (0, 1), (1, 1), (0, 0)]
    for op in POSSIBLE_OPERATIONS:
        print(f"Operation: {OPERATION2STRING[op]}")
        for p in pairs:
            print("\tInpt:", p, "- Outpt:", op(p[0], p[1]))
        print()

    n_pairs = 2
    n_samples = 5
    pairs, operations, outputs = generate_logical_task(n_pairs=n_pairs, n_samples=n_samples)
    input_strings, output_strings = convert_to_string_dataset(pairs, outputs)
    for i in range(len(pairs)):
        print(f"Sample {i+1}")
        print(f"\tPairs: {pairs[i]}")
        print(f"\tOperations: {[OPERATION2STRING[o] for o in operations]}")
        print(f"\tInput : {input_strings[i]}")
        print(f"\tOutput: {output_strings[i]}")
        print()

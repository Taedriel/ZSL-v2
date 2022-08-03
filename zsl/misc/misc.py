import logging
from typing import List, Dict, Callable

__all__ = ["dict2csv", "sim2dist", "print_mat"]

def dict2csv(filename : str, embeddings : Dict[str, List[float]]) -> None:
    """ write a dict of embeddings under a .CSV file

    the .CSV file is construct with a header looking like this :
    \tembeddings\t | 0 | 1 | 2 | 3 | ...
    where each line contain an embeddings for the word in the first row
    Args:
        filename (str) : a path to the file where the .csv is to be written
        embeddings (Dict[str, List[float]]): a dictionnary of embeddings

    """
    logging.info(f"writing dict to {filename} file")
    try:
        f = open(filename, "w")
    except OSError:
        raise OSError("Could not open file")

    dimension_number = len(next(iter(embeddings.values())))
    with f:
        print("embeddings", *[str(i) for i in range(dimension_number)], sep=",", file=f)
        for tag, embedding in embeddings.items():
            print(tag, *list(map(lambda x: str(float(x)), embedding)), sep=",", file=f)
    logging.info("done")


def sim2dist(mat : List[List[float]], func : Callable[[float], float] \
             = lambda x: 1 - x, hollow : bool = True) -> List[List[float]]:
    """ map the function func to each elements in the matrix

    apply the lambda function func to each element of the matrix. if hollow is set 
    to True, set the diagonal of the matrix to 0.
    Args:
        mat (List[List[float]]) : a matrix of number
        func (Callable[[float], float]) : a simple function to apply to each elem of the matrix
        hollow (bool) : whether to consider the diagonal of the matrix or not
    
    """
    logging.info(f"converting similarity matrix to distance matrix")
    inv_data = [[0 for i in range(len(mat[0]))] for j in range(len(mat))]

    for i, elem in enumerate(mat):
        for j, case in enumerate(elem):
            if i == j and hollow: 
                inv_data[i][j] = 0
            else:
                inv_data[i][j] = func(case)
                
    logging.info("done")
    return inv_data

def print_mat(mat : List[List[float]], format_function : Callable[[float], str]=lambda x: x) -> None:
    """ print a matrice on stdout

    format each number in the matrice using the format_function
    Args:
        mat (List[List[float]]) : a matrix of number
        format_function (Callable[[float], str]) : a simple format function to display numbers froms the matrix
    """
    for line in mat:
        for case in line:
            print(f"{format_function(case):8}", end="")
        print()

import os
import pandas as pd
import numpy as np


def reshape_sudoku(sgrid):
    grid = np.array([int(c) for c in sgrid])
    grid = grid.reshape(9, 9, 1)
    return grid


def reformat_input(input_csv):
    df = pd.read_csv(input_csv)
    x = df['quizzes'].to_numpy()
    y = df['solutions'].to_numpy()

    x = np.array([reshape_sudoku(l) for l in x])
    y = np.array([reshape_sudoku(l) for l in y])
    y2 = y-1
    np.save('x_all', x)
    np.save('y_all', y)
    np.save('y_all9', y2)


reformat_input('sudoku.csv')

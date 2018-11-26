import numpy as np
nrow=4
ncol=4
pitch={}
def find_row_col(row, col,action):
    if action == 0:  # left
        col = max(col - 1, 0)
    elif action == 1:  # down
        row = min(row + 1, nrow - 1)
    elif action == 2:  # right
        col = min(col + 1, ncol - 1)
    elif action == 3:  # up
        row = max(row - 1, 0)
    return (row, col)


if __name__ == '__main__':
    # a = []
    # dist={}
    # for i in range(0,nrow):
    #     for j in range(0,ncol):
            #pitch[i,j] = i*4+j

    for i in range(0,4):
        for j in range(0,4):
            for action in range(0,4):
                r,c=find_row_col(i, j,action)
                print('Row {} Column {} Action {} new_Row {} new_Column {}'.format(i,j,action,r,c))

import numpy as np

def cost1(A):
    """Penalizes difference in neighbouring fields. 
    Length 1 cross shaped.
      +   
    + o +
      +  
    """
    A = np.pad(A, pad_width=2)
    loss_matrix = np.zeros(A.shape)
    for shift in [(1,0), (-1,0), (0,1), (0,-1)]:
        loss_matrix += A*(A^np.roll(A, shift=shift, axis=(1,0)))
    return loss_matrix.sum()


def cost2(A):
    """Penalizes difference in horizontally neighbouring fields. 
    Rewards difference in vertically neighbouring fields. 
    Length 1 cross shaped. Applies only to 'ones'.
      -   
    + o +
      -  
    """
    A = np.pad(A, pad_width=2)
    loss_matrix = np.zeros(A.shape)
    for shift in [(1,0), (-1,0)]:
        loss_matrix += A*(A^np.roll(A, shift=shift, axis=(1,0)))

    for shift in [(0,1), (0,-1)]:
        loss_matrix -= A*(A^np.roll(A, shift=shift, axis=(1,0)))
    return loss_matrix.sum()


def cost3(A):
    """Penalizes tight clustering. 
    Length 1 box shaped. Applies to all fields.
    + + + 
    + o +
    + + +
    """
    A = np.pad(A, pad_width=2)
    loss_matrix = np.zeros(A.shape)
    for shift in [(i, j) for i in [-1,0,1] for j in [-1,0,1]]:
        loss_matrix += np.roll(A, shift=shift, axis=(1,0))

    loss_matrix = np.square(loss_matrix)
    return loss_matrix.sum()


def cost4(A):
    """Rewards diagonal chains harshly penalizes horizontal and vertical chains.
    -   +   -
      - + -  
    + + o + +
      - + -  
    -   +   -
    """
    A = np.pad(A, pad_width=2)
    penalty = np.zeros(A.shape)
    reward = np.zeros(A.shape)
    for shift in [(i, 0) for i in [-2, -1, 1, 2]]:
        penalty += np.roll(A, shift=shift, axis=(1,0))
    for shift in [(0, i) for i in [-2, -1, 1, 2]]:
        penalty += np.roll(A, shift=shift, axis=(1,0))
    for shift in [(i, i) for i in [-2, -1, 1, 2]]:
        reward += np.roll(A, shift=shift, axis=(1,0))
    for shift in [(i, -i) for i in [-2, -1, 1, 2]]:
        reward += np.roll(A, shift=shift, axis=(1,0))

    loss_matrix = np.square(A*penalty) - A*reward
    return loss_matrix.sum()

def cost5(A):
    """Harshly penalizes close neighbours, rewards mid range neighbours.
    - - - - -
    - + + + - 
    - + o + -
    - + + + -
    - - - - -
    """
    A = np.pad(A, pad_width=2)
    penalty = np.zeros(A.shape)
    reward = np.zeros(A.shape)
    for shift in [(i, j) for i in [-1,0,1] for j in [-1,0,1]]:
        penalty += np.roll(A, shift=shift, axis=(1,0))

    for shift in [(i, j) for i in [-2,2] for j in [-2,-1,0,1,2]]:
        reward += np.roll(A, shift=shift, axis=(1,0))

    for shift in [(i, j) for i in [-1,0,1] for j in [-2,2]]:
        reward += np.roll(A, shift=shift, axis=(1,0))

    loss_matrix = np.square(A*penalty) - A*reward
    return loss_matrix.sum()


def cost6(A):
    """This thing
      + - +  
    -   +   - 
    + - o - +
    -   +   -
      + - +   
    """
    A = np.pad(A, pad_width=2)
    penalty = np.zeros(A.shape)
    reward = np.zeros(A.shape)

    for shift in [(1,0),(2,-1),(2,1),(0,2)]:
        penalty += np.roll(A, shift=shift, axis=(1,0))
        penalty += np.roll(A, shift=(-shift[0], -shift[1]), axis=(1,0))
    for shift in [(1,0),(2,-1),(2,1),(0,2)]:
        reward += np.roll(A, shift=shift[::-1], axis=(1,0))
        reward += np.roll(A, shift=(-shift[1], -shift[0]), axis=(1,0))


    loss_matrix = np.square(A*penalty) - np.square(A*reward)
    return loss_matrix.sum()


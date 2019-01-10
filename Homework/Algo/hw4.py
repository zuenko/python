def printSolution(p, n):
        if p[n] == 1:
            k = 1
        else:
            k = printSolution(p,p[n]-1 ) + 1
        return k


def solveWordWrap(lens, n, L):
    INF = 9999
    
    extras = [[0 for i in range(n + 1)] for i in range(n + 1)]

    lc = [[0 for i in range(n + 1)] for i in range(n + 1)]

    c = [0 for i in range(n + 1)]

    p = [0 for i in range(n + 1)]

    # calculate extra spaces in a single
    # line. The value extra[i][j] indicates
    # extra spaces if words from word number
    # i to j are placed in a single line
    for i in range(n + 1):
        extras[i][i] = L - lens[i - 1]
        for j in range(i + 1, n + 1):
            extras[i][j] = (extras[i][j - 1] - lens[j - 1] - 1)
            if extras[i][j] < 0:
                lc[i][j] = INF
            else:
                lc[i][j] = (extras[i][j]**2)

    # Calculate minimum cost and find
    # minimum cost arrangement. The value
    # c[j] indicates optimized cost to
    # arrange words from word number 1 to j.
    c[0] = 0
    for j in range(1, n + 1):
        c[j] = INF
        for i in range(1, j + 1):
            if (c[i - 1] != INF 
                and lc[i][j] != INF 
                and ((c[i - 1] + lc[i][j]) < c[j])):
                c[j] = c[i - 1] + lc[i][j]
                p[j] = i

    error = str(c[-1])
    k = str(printSolution(p, n))
    return error, k

if __name__ == "__main__":
    with open('input.txt', "r") as f:
        params = f.read().splitlines()

    L = int(params[0])
    lens = [len(i) for i in params[1].split(' ')]
    n = len(lens)
    
    error, k = solveWordWrap(lens, n, L)

    with open('output.txt', "w") as f:
        f.write(error + '\n' + k)
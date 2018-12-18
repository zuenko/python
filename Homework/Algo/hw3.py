def check_rectangle(point, rect):
    x, y = point
    # bottom-left, top-right
    x1,y1, x2,y2 = rect[0][0], rect[1][1], rect[1][0], rect[0][1]        
    
    if x1 > x2 and y1 > y2:
        tmp = x2, y2
        x2, y2 = x1, x2
        x1, y1 = tmp    
    return (x >= x1 and x <= x2 and y >= y1 and y <= y2)

def calculation(queries, mines):
    result = []
    for querie in queries:
        check_true = []
        k = 0
    
        for rect in queries[querie]:
            for cord in mines:
                # print (querie, cord, cord not in check_true)
                if (check_rectangle(cord[0], rect) == True) and cord not in check_true:
                    check_true.append(cord)
                    k += cord[1]
                    #print (querie, cord, rect)
    
        result.append(k)
    return result

if __name__ == "__main__":
    f     = open('input.txt',  'r')
    f_out = open('output.txt', 'w')
    
    n_mines = int(f.readline().split()[0])
    
    # build dict of mines
    mines = []
    for i in range(n_mines):
        tmp = f.readline().split()
        mines.append(((int(tmp[0]), int(tmp[1])), int(tmp[2]), i))
    
    n_queries = int(f.readline().split()[0])
    # build dict of queries
    queries = {}
    for i in range(n_queries):
        n_rectangles = int(f.readline().split()[0])
        
        l_queries = []
        for j in range(n_rectangles):
            tmp = f.readline().split()
            l_queries.append(([int(tmp[0]), int(tmp[1])], [int(tmp[2]), int(tmp[3])]))
        
        queries[i] = l_queries
    

    answer = calculation(queries, mines)
    # write result
    for itr, item in enumerate(answer):
        if len(answer) - 1 != itr:
            f_out.write("%s\n" % item)
        else:
            f_out.write("%s" % item)

    
    # close
    f.close 
    f_out.close
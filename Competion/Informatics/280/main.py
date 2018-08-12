def isLineCross(line1, line2):
    x11, y11, x12, y12 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    x21, y21, x22, y22 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
    
    # Swap if ascending
    if (x12 < x11):
        tmp1, tmp2 = x11, x12
        x11, x12 = tmp2, tmp1
    if (x22 < x21):
        tmp1, tmp2 = x21, x22
        x11, x12 = tmp2, tmp1
    
    # Check existing of potentional interval for the points of lines    
    if (x12 < x21):
        return False # They ain't got no mutual x
    
    # If they both are vertical
    if (x11 - x12) == 0 and (x21 - x22) == 0:
        # If they lies on the same x
        if (x11 == x21):
            # Check if they intersept. Do they have shaerd Y
            if (not max(y11, y12) < min(y21, y22)) or \
               (min(y11, y12) > max(y21, y22)):
                    return True
        return False
    
    # Lets find the coefficients of equations
    # f1(x) = A1*x + b1 = y
    # f2(x) = A2*x + b2 = y
    # If first line is vertical
    if (x11 - x12) == 0:
        # Lets Find Xa, Ya - points of intersection of two points
        Xa = x11
        A2 = (y21 - y22)/(x21 - x22)
        b2 = y21 - A2 * x21
        Ya = A2 * Xa + b2
        if (x21 <= Xa and x22 >= Xa and min(y11, y12) <= Ya and max(y11, y12) >= Ya):
            return True
        
        return False
    
    # If second line are vertical
    if (x21 - x22) == 0:
        # Lets Find Xa, Ya - points of intersection of two points
        Xa = x21
        A1 = (y11 - y12)/(x11 - x12)
        b1 = y11 - A1 * x11
        Ya = A1 * Xa + b1
        if (x11 <= Xa and x12 >= Xa and min(y21, y22) <= Ya and max(y21, y22) >= Ya):
            return True
        
        return False
    # If they both are non-vertical
    A1 = (y11 - y12)/(x11 - x12)
    A2 = (y21 - y22)/(x21 - x22)
    b1 = y11 - A1 * x11
    b2 = y21 - A2 * x21
    
    if A1 == A2:
        return True # They are paralel
    
    # Xa - x point of their instersection
    Xa = (b2 - b1)/(A1 - A2)
    
    if (Xa < max(x11, x21) or Xa > min(x12, x22)):
        return False # Point Xa - not inside inseresection proection
    else:
        return True

if __name__ == "__main__":
	with open("input.txt") as f: 
		data = f.readlines()
	for i in range(len(data)):
		data[i] = [int(n) for n in data[i].split()]

	line1 = [(data[0][0], data[0][1]), (data[0][2], data[0][3])]
	line2 = [(data[1][0], data[1][1]), (data[1][2], data[1][3])]

	with open("output.txt", 'w') as f:
		if isLineCross(line1, line2):
			f.write('YES')
		else:
			f.write('NO')
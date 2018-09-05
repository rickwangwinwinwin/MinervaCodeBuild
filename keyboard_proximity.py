# #!/usr/bin/python 05092018

SHIFT_COST = 3.0
INSERTION_COST = 1.0
DELETION_COST = 1.0
SUBSTITUTION_COST = 1.0

qwertyKeyboardArray = [
    ['`','1','2','3','4','5','6','7','8','9','0','-','='],
    ['q','w','e','r','t','y','u','i','o','p','[',']','\\'],
    ['a','s','d','f','g','h','j','k','l',';','\''],
    ['z','x','c','v','b','n','m',',','.','/'],
    ['', '', ' ', ' ', ' ', ' ', ' ', '', '']
    ]

qwertyShiftedKeyboardArray = [
    ['~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?'],
    ['', '', ' ', ' ', ' ', ' ', ' ', '', '']
    ]

layoutDict = {'QWERTY': (qwertyKeyboardArray, qwertyShiftedKeyboardArray)}

# # Sets the default keyboard to use to be QWERTY.
keyboardArray = qwertyKeyboardArray
shiftedKeyboardArray = qwertyShiftedKeyboardArray


# # Returns the keyboard layout c "lives in"; for instance, if c is A, this will return the shifted keyboard array, but if it is a, it will return the regular
# # keyboard array.  Raises a ValueError if character is in neither array
def arrayForChar(c):
    if (True in [c in r for r in keyboardArray]):
        return keyboardArray
    elif (True in [c in r for r in shiftedKeyboardArray]):
        return shiftedKeyboardArray
    else:
        raise ValueError(c + " not found in any keyboard layouts")

# # Finds a 2-tuple representing c's position on the given keyboard array.  If the character is not in the given array, throws a ValueError
def getCharacterCoord(c, array):
    row = -1
    column = -1
    for r in array:
        if c in r:
            row = array.index(r)
            column = r.index(c)
            return (row, column)
    raise ValueError(c + " not found in given keyboard layout")

# # Finds the Euclidean distance between two characters, regardless of whether
# # they're shifted or not.
def euclideanKeyboardDistance(c1, c2):
    coord1 = getCharacterCoord(c1, arrayForChar(c1))
    coord2 = getCharacterCoord(c2, arrayForChar(c2))
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**(0.5)

# # The cost of inserting c at position i in string s
def insertionCost(s, i, c):
    if not s or i >= len(s):
        return INSERTION_COST
    cost = INSERTION_COST
    if arrayForChar(s[i]) != arrayForChar(c):
        # We weren't holding down the shift key when we were typing the original string, but started holding it down while inserting this character, or
        # vice versa.  Either way, this action should have a higher cost.
        cost += SHIFT_COST
    cost += euclideanKeyboardDistance(s[i], c)
    return cost

# The cost of omitting the character at position i in string s
def deletionCost(s, i):
    return DELETION_COST

# # The cost of substituting c at position i in string s
def substitutionCost(s, i, c):
    cost = SUBSTITUTION_COST
    if len(s) == 0 or i >= len(s):
        return INSERTION_COST
    if arrayForChar(s[i]) != arrayForChar(c):
        # We weren't holding down the shift key when we were typing the original string, but started holding it down while inserting this character, or
        # vice versa.  Either way, this action should have a higher cost.
        cost += SHIFT_COST
    cost += euclideanKeyboardDistance(s[i], c)
    return cost

# # Finds the typo distance (a floating point number) between two strings, based on the canonical Levenshtein distance algorithm.
def typoDistance(s, t, layout='QWERTY'):
    if layout in layoutDict:
        keyboardArray, shiftedKeyboardArray = layoutDict[layout]
    else:
        raise KeyError(layout + " keyboard layout not supported")
    
    # A multidimensional array of 0s with len(s) rows and len(t) columns.
    d = [[0]*(len(t) + 1) for i in range(len(s) + 1)]

    for i in range(len(s) + 1):
        d[i][0] = sum([deletionCost(s, j - 1) for j in range(i)])
    for i in range(len(t) + 1):
        intermediateString = ""
        cost = 0.0
        for j in range(i):
            cost += insertionCost(intermediateString, j - 1, t[j - 1])
            intermediateString = intermediateString + t[j - 1]
        d[0][i] = cost

    for j in range(1, len(t) + 1):
        for i in range(1, len(s) + 1):
            if s[i - 1] == t[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                delCost = deletionCost(s, i - 1)
                insertCost = insertionCost(s, i, t[j - 1])
                subCost = substitutionCost(s, i - 1, t[j - 1])
                d[i][j] = min(d[i - 1][j] + delCost,
                              d[i][j - 1] + insertCost,
                              d[i - 1][j - 1] + subCost)

    return d[len(s)][len(t)]


        
#print(typoDistance('Bhavana','Bhuvana'))



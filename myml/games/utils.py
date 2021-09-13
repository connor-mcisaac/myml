_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def _number_to_letters(x):
    n = 1
    num = 0
    while x >= len(_letters) ** n + num:
        num += len(_letters) ** n
        n += 1
    x -= num
    letters = ''
    for i in range(n, 0, -1):
        num = x // len(_letters) ** (i - 1)
        letters += _letters[num]
        x = x % len(_letters) ** (i - 1)
    return letters

def _letters_to_number(letters):
    letters = letters.upper()
    x = 0
    for i in range(1, len(letters)):
        x += len(_letters) ** i
    for i in range(len(letters)):
        x += (_letters.index(letters[i])
              * len(_letters) ** (len(letters) - i - 1))
    return x

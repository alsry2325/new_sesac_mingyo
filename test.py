n = 2
array = [
    [1, 2],
    [4, 10]
]

total = 0
for i in range(n):
    for j in range(n):
        if array[i][j] % 2 == 0:
            total += array[i][j]
            print(array[i][j])


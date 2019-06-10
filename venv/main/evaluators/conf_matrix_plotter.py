from seaborn import heatmap
from pandas import DataFrame
import matplotlib.pyplot as plt

# arra = [[376, 41, 5, 17, 61],
#         [10, 453, 2, 3, 32],
#         [20, 19, 357, 18, 86],
#         [8, 9, 8, 456, 19],
#         [16, 15, 8, 3, 458]]

# arra = [[372, 42, 27, 6, 53],
# [35, 395, 30, 9, 31],
# [37, 16, 380, 11, 56],
# [75, 21, 63, 323, 18],
# [31, 18, 21, 1, 429]]

# arra = [[384, 17, 11, 9, 79],
# [49, 395, 2, 6, 74],
# [41, 12, 326, 15, 106],
# [12, 9, 31, 425, 23],
# [22, 11, 8, 1, 458]]


arra = [[397, 9, 1, 12, 81],
[122, 299, 2, 17, 60],
[54, 11, 262, 37, 136],
[55, 8, 17, 369, 51],
[39, 9, 2, 7, 443]]


classes = ["Audi A4", "BMW 3", "VW Golf V", "Mercedes E", "Opel Astra"]

df = DataFrame(arra, index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (10,7))
heatmap(df, annot=True)
plt.show()

# [[372, 42, 27, 6, 53],
# [35, 395, 30, 9, 31],
# [37, 16, 380, 11, 56],
# [75, 21, 63, 323, 18],
# [31, 18, 21, 1, 429]]

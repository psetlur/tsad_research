wd = {'1': {'2': {'3': '4', '4': '5'}, '3': {'3': '4', '4': '5'}},
      '2': {'2': {'3': '4', '4': '5'}, '3': {'3': '4', '4': '5'}}}

with open(f'1.txt', 'w') as file:
    file.write('wd: ' + str(wd))

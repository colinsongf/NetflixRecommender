import csv

with open('test.csv', 'rt') as file :
  data = csv.reader(file)
  dataset = list(data)
  for x in range(len(dataset)) :
    filename = 'mv_' + dataset[x][0] + '.txt'
    with open(filename, 'rt') as file2 :
      data2 = csv.reader(file2)
      dataset2 = list(data)
      print(filename)
      for y in range(len(dataset2)) :
        if dataset2[y][0] == dataset[x][1] and dataset2[y][2] == dataset[x][3] :
          dataset[x][2] = dataset2[y][1]
  with open('test2.csv', 'w') as file3 :
    writer = csv.writer(file3, delimiter=',')
    writer.writerows(dataset)

import random
import numpy as np
import matplotlib.pyplot as plt


class Bee:
    def __init__(self, tour):
        self.tour = tour
        self.distance = 0

    def compute_distance(self, dist_matrix):
        self.distance = sum([dist_matrix[self.tour[i - 1]][self.tour[i]] for i in range(1, len(self.tour))])
        self.distance += dist_matrix[self.tour[-1]][self.tour[0]]


class ABC:
    def __init__(self, dist_matrix, num_bees, max_iter, num_onlooker_bees):
        self.dist_matrix = dist_matrix
        self.num_bees = num_bees
        self.max_iter = max_iter
        self.num_onlooker_bees = num_onlooker_bees
        self.bees = self.init_bees()

    def init_bees(self):
        bees = []
        for _ in range(self.num_bees):
            tour = list(range(len(self.dist_matrix)))
            random.shuffle(tour)
            bee = Bee(tour)
            bee.compute_distance(self.dist_matrix)
            bees.append(bee)
        return bees

    def send_onlooker_bees(self):
        for _ in range(self.num_onlooker_bees):
            bee = random.choice(self.bees)
            new_tour = self.mutate(bee.tour)
            new_bee = Bee(new_tour)
            new_bee.compute_distance(self.dist_matrix)
            if new_bee.distance < bee.distance:
                self.bees.remove(bee)
                self.bees.append(new_bee)

    def mutate(self, tour):
        new_tour = tour[:]
        i, j = random.sample(range(len(new_tour)), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def solve(self):
        for _ in range(self.max_iter):
            self.send_onlooker_bees()
        best_bee = min(self.bees, key=lambda bee: bee.distance)
        return best_bee.tour, best_bee.distance


def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data


data = read_tsp('data/st70.tsp')

data = np.array(data)
data = data[:, 1:]
# 加上一行因为会回到起点
show_data = np.vstack([data, data[0]])

num_city = len(data)
location = data


def compute_dis_mat(num_city, location):
    dis_mat = np.zeros((num_city, num_city))
    for i in range(num_city):
        for j in range(num_city):
            if i == j:
                dis_mat[i][j] = np.inf
                continue
            a = location[i]
            b = location[j]
            tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
            dis_mat[i][j] = tmp
    return dis_mat


dist_matrix = compute_dis_mat(len(data), data)
aco = ABC(dist_matrix, len(data), len(data) * 100, len(data) // 2)
Best_path, Best = aco.solve()
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)

Best_path_pos_x = []
Best_path_pos_y = []

for elem in Best_path:
    Best_path_pos_x.append(data[elem][0])
    Best_path_pos_y.append(data[elem][1])
axs[0].scatter(Best_path_pos_x, Best_path_pos_y)
Best_path_pos_x.append(data[Best_path[0]][0])
Best_path_pos_y.append(data[Best_path[0]][1])

axs[0].plot(Best_path_pos_x, Best_path_pos_y)
axs[0].set_title('规划结果')
# axs[1].plot(iterations, best_record)
# axs[1].set_title('收敛曲线')
plt.show()

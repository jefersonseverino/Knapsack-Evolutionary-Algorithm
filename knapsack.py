import random

POPULATION_SIZE = 300
GENOMA_SIZE = 100
NUM_OF_ITEMS = 100
MUTATION_PROB = 0.02
CROSSOVER_PROB = 0.75
MAX_GENERATIONS = 500
NUM_OF_PARENTS = POPULATION_SIZE // 2

items_list = []
class Item:

  def __init__(self, value, weight):
    self.value = value
    self.weight = weight


def initialize_problem():
  for _ in range(GENOMA_SIZE):
    value = random.randint(1, 1000)
    weight = random.randint(1, 50)

    items_list.append(Item(value, weight))

  return items_list

def initialize_population():
  population = []
  for _ in range(POPULATION_SIZE):
    individual = [0] * GENOMA_SIZE
    total_weight = 0
    while total_weight <= 100:
      random_item = random.randint(0, NUM_OF_ITEMS-1)
      if total_weight + items_list[random_item].weight > 100:
        break
      total_weight += items_list[random_item].weight
      individual[random_item] = 1

    population.append(individual)

  return population

def fitness(genome):
  value = 0
  weight = 0

  for idx, gene in enumerate(genome):
    value += gene * items_list[idx].value
    weight += gene * items_list[idx].weight

  fitness = value

  if weight > 100:
    fitness = 0

  return fitness

def check_stop_condition(population, best_fit, best_fit_count):
  fitness_list = [fitness(individual) for individual in population]
  current_best = sum(fitness_list)

  if abs(current_best - best_fit) < 300:
    best_fit_count += 1
  
  elif current_best > best_fit:
    best_fit_count = 0
    best_fit = current_best

  if best_fit_count > 10:
    print('Convergiu')
    return True, best_fit, best_fit_count
  
  return False, best_fit, best_fit_count

def tournament(population):
  parents_candidates = random.sample(population, 5)
  parents_candidates = sorted(parents_candidates, reverse=True, key=fitness)
  return parents_candidates[:2]

def parents_selection(population):
  parents = []
  for _ in range(0, NUM_OF_PARENTS):
    parents.append(tournament(population))

  return parents

def crossover(parents_list):
  children = []
  for parents in parents_list:
    idx = random.randint(0, POPULATION_SIZE - 1)
    children.append(parents[0][:idx] + parents[1][idx:])
    children.append(parents[1][:idx] + parents[0][idx:])
  return children

def mutation(population):
  for individual in population:
    for i in range(GENOMA_SIZE):
      if random.random() < MUTATION_PROB:
          individual[i] = 1 - individual[i]

  return population

def roulette(population):
  fitness_list = [fitness(individual) for individual in population]
  total_fitness = sum(fitness_list)
  probabilities = [(ind_fitness / total_fitness) for ind_fitness in fitness_list]
  selected_individuals = random.choices(population, probabilities, k=POPULATION_SIZE)

  return selected_individuals

def find_solution(population):
  num_iterations = 0
  best_fit_count = 0
  best_fit = max([fitness(individual) for individual in population])

  for num_iterations in range(MAX_GENERATIONS):

    parents = parents_selection(population)
    children = parents

    if random.random() < CROSSOVER_PROB:
      children = crossover(parents)
      children = mutation(children)
      population += children

    population = roulette(population)

    stop, best_fit, best_fit_count = check_stop_condition(population, best_fit, best_fit_count)
    if stop:
      break
  
  return population, num_iterations

def knapsack_dp(items_list, capacity):
    n = len(items_list)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            current_item = items_list[i - 1]
            if current_item.weight > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - current_item.weight] + current_item.value)
    selected_items = []
    total_value = dp[n][capacity]
    total_weight = capacity
    for i in range(n, 0, -1):
        if dp[i][total_weight] != dp[i - 1][total_weight]:
            selected_items.append(i - 1)
            total_weight -= items_list[i - 1].weight
    selected_items = selected_items[::-1]

    return selected_items, total_value, capacity-total_weight

def analyze_solution(population, num_iterations):
  fitness_list = [fitness(individual) for individual in population]

  idx_solution = fitness_list.index(max(fitness_list))
  best_individual = population[idx_solution]
  choosen_items_idx = [i for i, valor in enumerate(best_individual) if valor == 1]

  total_value = sum(items_list[i].value for i in choosen_items_idx)
  total_weight = sum(items_list[i].weight for i in choosen_items_idx)

  dp_choosen_items_idx, dp_total_value, dp_total_weight = knapsack_dp(items_list, 100)

  print('Solução - Algoritmo Genético')
  print('Número de iterações:', num_iterations)
  print('Itens escolhidos:', choosen_items_idx)
  print('Valor total:', total_value)
  print('Peso total:', total_weight)

  print('\n-------------------------\n')

  print('Solução - DP')
  print('Itens escolhidos:', dp_choosen_items_idx)
  print('Valor total:', dp_total_value)
  print('Peso total:', dp_total_weight)


if __name__ == '__main__':
  initialize_problem()
  initial_population = initialize_population()
  population, num_iterations = find_solution(initial_population)
  analyze_solution(population, num_iterations)

import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


class gameOb():
	def __init__(self, coordinates, size, intensity, channel, reward, name):
		self.x = coordinates[0]
		self.y = coordinates[1]
		self.size = size
		self.intensity = intensity
		self.channel = channel
		self.reward = reward
		self.name = name


class gameEnv():
	def __init__(self, partial, size):
		self.sizeX = size
		self.sizeY = size
		self.actions = 4
		self.objects = []
		self.partial = partial
		self.state = None
		a = self.reset()
		plt.imshow(a, interpolation="nearest")

	def reset(self):
		self.objects = []
		self.objects.append(gameOb(self.newPosition(), 1, 1, 2, None, "hero"))
		self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, "goal"))
		self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, "fire"))
		self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, "goal"))
		self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, "fire"))
		self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, "goal"))
		self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, "goal"))
		self.state = self.renderEnv()
		return self.state

	def moveChar(self, direction):
		UP = 0
		DOWN = 1
		LEFT = 2
		RIGHT = 3

		hero = self.objects[0]
		heroX = hero.x
		heroY = hero.y

		penalize = 0.

		if direction == UP and hero.y >= 1:
			hero.y -= 1
		elif direction == DOWN and hero.y <= self.sizeY - 2:
			hero.y += 1
		elif direction == LEFT and hero.x >= 1:
			hero.x -= 1
		elif direction == RIGHT and hero.x <= self.sizeX - 2:
			hero.x += 1

		if hero.x == heroX and hero.y == heroY:
			penalize = 0.0

		self.objects[0] = hero
		return penalize

	def newPosition(self):
		iterables = [range(self.sizeX), range(self.sizeY)]
		points = []

		for t in itertools.product(*iterables):
			points.append(t)

		currentPositions = []
		for objectA in self.objects:
			if (objectA.x, objectA.y) not in currentPositions:
				currentPositions.append((objectA.x, objectA.y))

		for pos in currentPositions:
			points.remove(pos)

		location = np.random.choice(range(len(points)), replace=False)
		return points[location]

	def checkGoal(self):
		goals = []
		for obj in self.objects:
			if obj.name == "hero":
				hero = obj
			else:
				goals.append(obj)

		for goal in goals:
			if hero.x == goal.x and hero.y == goal.y:
				self.objects.remove(goal)

				if goal.reward == 1:
					self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, "goal"))
				else:
					self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, "fire"))
				return goal.reward, False

		return 0.0, False

	def renderEnv(self):
		# a = np.zeros([self.sizeY, self.sizeX, 3])
		a = np.ones([self.sizeY + 2, self.sizeX + 2, 3])

		a[1:-1, 1:-1, :] = 0
		hero = None

		for item in self.objects:
			a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity

			if item.name == "hero":
				hero = item

		if self.partial is True:
			a = a[hero.y:hero.y + 3, hero.x:hero.x + 3, :]

		b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp="nearest")
		c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp="nearest")
		d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp="nearest")
		a = np.stack([b, c, d], axis=2)

		return a

	def step(self, action):
		penalty = self.moveChar(action)
		reward, done = self.checkGoal()
		state = self.renderEnv()

		if reward is None:
			print(done)
			print(reward)
			print(penalty)

		return state, (reward + penalty), done

from __future__ import print_function
import sys
import random
from distanceCalculator import Distancer
from game import Actions
import time as time
import os
import numpy as np
import math
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
from wekaI import Weka


class NullGraphics(object):
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True, elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.countActions = 0

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution()
                             for inf in self.inferenceModules]
        self.firstMove = True

    #def observationFunction(self, gameState):
        # Removes the ghost states from the gameState
        # agents = gameState.data.agentStates
        # gameState.data.agentStates = [agents[0]] + \
        #     [None for i in range(1, len(agents))]
        # return gameState
        
    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        self.countActions = 1 + self.countActions
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions,
              " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ",
              gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ",
              gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        map = gameState.getWalls()
        print(map)
        # Score
        print("Score: ", gameState.getScore())

    def printLineData(self, gameState):
        # Dimensiones del tablero
        width, height = gameState.data.layout.width, gameState.data.layout.height
        # Coordenadas de pacman
        pacman_position = gameState.getPacmanPosition()
        # Acciones legales de pacman
        legal_actions = gameState.getLegalPacmanActions()
        # Comprobando cuales de las acciones legales estan incluidas
        legal_N = 'North' in legal_actions
        legal_S = 'South' in legal_actions
        legal_E = 'East' in legal_actions
        legal_W = 'West' in legal_actions
        # Direcciones de pacman
        pacman_dir = gameState.data.agentStates[0].getDirection()
        # Agentes vivos
        living_ghosts = gameState.getLivingGhosts()
        # Posiciones de los fantasmas
        ghosts_positions = gameState.getGhostPositions()
        # Numero de agentes
        num_ghosts = gameState.getNumAgents() - 1

        # Direcciones de los fantasmas
        ghosts_directions = [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)]

        ghosts_distances = gameState.data.ghostDistances

        # Movimientos verticales
        mov_ver = 999999
        # Movimientos horizontales
        mov_hor = 999999
        # Movimientos a cada distancias
        min_distancia_norte = 999999
        min_distancia_sur = 999999
        min_distancia_oeste = 999999
        min_distancia_este = 999999
        # Variable spara guardar el número de fantasmas en cierta dirección
        num_ghost_N = 0
        num_ghost_S = 0
        num_ghost_W = 0
        num_ghost_E = 0
        # Variables para guardar las direcciones de los fantasmas más cercanos en cada dirección
        direccion_fantasma_N = 'None'
        direccion_fantasma_S = 'None'
        direccion_fantasma_W = 'None'
        direccion_fantasma_E = 'None'
        direccion_fantasma_cercano = "None"
        # Variable para la distancia al fantasma más cercano
        distancia_cercana_fantasma = 9999999
        # Variable para guardar los fantasmas no comidos
        num_living_ghosts = 0

        # numero de fantasmas mirando en cada direccion
        num_ghost_ori_N = 0
        num_ghost_ori_S = 0
        num_ghost_ori_E = 0
        num_ghost_ori_W = 0

        # Posiciones verticales para estar en la misma fila que el fantasma más cercano, en caso de empate se elige siempre al primero
        for agente in range(0, num_ghosts):
            if ghosts_distances[agente] is not None and (abs(ghosts_positions[agente][0] - pacman_position[0]) < mov_ver):
                mov_ver = abs(ghosts_positions[agente][0] - pacman_position[0])
            if ghosts_distances[agente] is not None and (abs(ghosts_positions[agente][1] - pacman_position[1]) < mov_hor):
                mov_hor = abs(ghosts_positions[agente][1] - pacman_position[1])
            # Oeste
            if ghosts_distances[agente] is not None and (ghosts_positions[agente][0] < pacman_position[0]):
                num_ghost_W += 1
                if min_distancia_oeste > abs(ghosts_positions[agente][0] - pacman_position[0]):
                    min_distancia_oeste = abs(
                        ghosts_positions[agente][0] - pacman_position[0])
                    direccion_fantasma_W = ghosts_directions[agente]

            # Este
            if ghosts_distances[agente] is not None and (ghosts_positions[agente][0] > pacman_position[0]):
                num_ghost_E += 1
                if min_distancia_este > abs(ghosts_positions[agente][0] - pacman_position[0]):
                    min_distancia_este = abs(
                        ghosts_positions[agente][0] - pacman_position[0])
                    direccion_fantasma_E = ghosts_directions[agente]

            # Norte
            if ghosts_distances[agente] is not None and (ghosts_positions[agente][1] > pacman_position[1]):
                num_ghost_N += 1
                if min_distancia_norte > abs(ghosts_positions[agente][1] - pacman_position[1]):
                    min_distancia_norte = abs(
                        ghosts_positions[agente][1] - pacman_position[1])
                    direccion_fantasma_N = ghosts_directions[agente]
            # Sur
            if ghosts_distances[agente] is not None and (ghosts_positions[agente][1] < pacman_position[1]):
                num_ghost_S += 1
                if min_distancia_sur > abs(ghosts_positions[agente][1] - pacman_position[1]):
                    min_distancia_sur = abs(
                        ghosts_positions[agente][1] - pacman_position[1])
                    direccion_fantasma_S = ghosts_directions[agente]

            # Calculando la dirección del fantasma más cercano
            if distancia_cercana_fantasma > (abs(ghosts_positions[agente][1]-pacman_position[1])+abs(ghosts_positions[agente][0]-pacman_position[0])):
                distancia_cercana_fantasma = abs(
                    ghosts_positions[agente][1]-pacman_position[1])+abs(ghosts_positions[agente][0]-pacman_position[0])
                direccion_fantasma_cercano = ghosts_directions[agente]

            if living_ghosts[agente+1]:
                num_living_ghosts = num_living_ghosts+1

            if ghosts_directions[agente] == "North":
                num_ghost_ori_N = num_ghost_ori_N+1
            if ghosts_directions[agente] == "West":
                num_ghost_ori_W = num_ghost_ori_W+1
            if ghosts_directions[agente] == "South":
                num_ghost_ori_S = num_ghost_ori_S+1
            if ghosts_directions[agente] == "East":
                num_ghost_ori_E = num_ghost_ori_E+1

        # En el caso de que no quede comida, la distancia más cercana se cambia de None a -1
        distance_food_nearest = -1
        if gameState.getDistanceNearestFood() is not None:
            distance_food_nearest = gameState.getDistanceNearestFood()

        map = gameState.getWalls()

        distancia_pared_norte = 0
        distancia_pared_sur = 0
        distancia_pared_este = 0
        distancia_pared_oeste = 0

        found = False
        for y in range(pacman_position[1], height):
            if not found and not map[pacman_position[0]][y]:
                distancia_pared_norte += 1
            elif not found:
                found = True

        found = False
        for y in range(pacman_position[1], 0, -1):
            if not found and not map[pacman_position[0]][y]:
                distancia_pared_sur += 1
            elif not found:
                found = True

        found = False
        for x in range(pacman_position[0], width):
            if not found and not map[x][pacman_position[1]]:
                distancia_pared_este += 1
            elif not found:
                found = True

        found = False
        for x in range(pacman_position[0], 0, -1):
            if not found and not map[x][pacman_position[1]]:
                distancia_pared_oeste += 1
            elif not found:
                found = True

        vision = [[0 for x in range(3)] for y in range(3)]

        for x in range(0, 3):
            for y in range(0, 3):
                if not(x == 1 and y == 1):
                    vision[x][y] = map[pacman_position[0] +
                                       x - 1][pacman_position[1] + y - 1]

                    if not vision[x][y]:
                        for agente in range(0, num_ghosts):
                            if ghosts_positions[agente][0] == pacman_position[0] + x - 1 and ghosts_positions[agente][1] == pacman_position[1] + y - 1:
                                vision[x][y] = 'Ghost'

        # Linea para escribir en fichero, se han casteado a string todos los valores para evitar errores en la escritura.
        lineData = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},".format(
            str(self.countActions),
            str(width), str(height),
            str(pacman_position[0]), str(pacman_position[1]),
            str(legal_N), str(legal_S), str(legal_E), str(legal_W),
            str(pacman_dir),
            str(num_ghosts),
            str(num_living_ghosts),
            str(mov_ver),
            str(mov_hor),
            str(min_distancia_norte),
            str(min_distancia_sur),
            str(min_distancia_oeste),
            str(min_distancia_este),

            str(vision[0][0]),
            str(vision[0][1]),
            str(vision[0][2]),
            str(vision[1][0]),
            str(vision[1][2]),
            str(vision[2][2]),
            str(vision[2][1]),
            str(vision[2][2]),

            str(distancia_pared_norte),
            str(distancia_pared_sur),
            str(distancia_pared_este),
            str(distancia_pared_oeste),


            str(num_ghost_N),
            str(num_ghost_S),
            str(num_ghost_W),
            str(num_ghost_E),
            str(num_ghost_ori_N),
            str(num_ghost_ori_S),
            str(num_ghost_ori_W),
            str(num_ghost_ori_E),
            str(direccion_fantasma_N),
            str(direccion_fantasma_S),
            str(direccion_fantasma_W),
            str(direccion_fantasma_E),
            str(direccion_fantasma_cercano),
            str(distancia_cercana_fantasma),
            str(gameState.getNumFood()),
            str(distance_food_nearest),
            str(gameState.getScore()))

        return lineData


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        self.printInfo(gameState)
        return KeyboardAgent.getAction(self, gameState)


'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + \
                    gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  # Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:
            move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal:
            move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
                                            if livingGhosts[i+1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + \
                    gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions,
              " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ",
              gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ",
              gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        map = gameState.getWalls()
        print(map)
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        legal = gameState.getLegalActions(0)  # Legal position from the pacman
        # Object to calculate the distances between positions
        distancer = Distancer(gameState.data.layout, False)

        # Getting the closest ghost in the layout
        oldDistance = 1000000000
        newDistance = -1
        nearestGhostIndex = 0

        # We iterate through every ghost in the game
        for i in range(len(gameState.data.ghostDistances)):
            # If the ghost hasn't been eaten yet
            if gameState.data.ghostDistances[i] is not None:
                # Calculating the distance between this ghost and Pac-Man
                newDistance = distancer.getDistance(
                    gameState.getPacmanPosition(), gameState.getGhostPositions()[i])
                # If the distance is lower than the minimum previosly found we save the index of the ghost
                if newDistance is not None and oldDistance is not None and oldDistance > newDistance:
                    oldDistance = newDistance
                    nearestGhostIndex = i

        # Selecting best action
        minDistance = -1
        bestAction = "Stop"
        # We iterate through every legal action
        for action in legal:
            # We obtain the position of Pac-Man and modify it depending on the current action
            pacmanPosition = gameState.getPacmanPosition()

            if action == "North":
                pacmanPosition = (pacmanPosition[0], pacmanPosition[1]+1)

            elif action == "East":
                pacmanPosition = (pacmanPosition[0]+1, pacmanPosition[1])

            elif action == "South":
                pacmanPosition = (pacmanPosition[0], pacmanPosition[1]-1)

            elif action == "West":
                pacmanPosition = (pacmanPosition[0]-1, pacmanPosition[1])

            # We calculate the distance between the new position of Pac-Man and the closest ghost
            newDistance = distancer.getDistance(
                pacmanPosition, gameState.getGhostPositions()[nearestGhostIndex])
            # If we are in the first iteration
            if minDistance == -1:
                minDistance = newDistance
                newDistance = minDistance
                bestAction = action
            # If the distance to the closest ghost is reduced we save the action
            elif newDistance < minDistance or newDistance is None:
                bestAction = action
                minDistance = newDistance

        nextState = gameState.generateSuccessor(0, bestAction)
        self.printInfo(nextState)
        return bestAction


class AutomaticAgent(BustersAgent):

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True, elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.countActions = 0
        self.weka = Weka()
        self.weka.start_jvm()

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + \
                    gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions,
              " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ",
              gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ",
              gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        map = gameState.getWalls()
        print(map)
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):

        x = self.getData(gameState)
        x = x.replace(",", "")
        x = x.split()
        a = self.weka.predict('./ibk3_tutorial_other.model',
                              x, "./training_tutorial1_no_score.arff")
        legal = gameState.getLegalPacmanActions()
        if a not in legal:
            a = random.choice(legal)

        nextState = gameState.generateSuccessor(0, a)
        self.printInfo(nextState)

        

        return a

    def getData(self, gameState):
        # Dimensiones del tablero
        width, height = gameState.data.layout.width, gameState.data.layout.height
        # Coordenadas de pacman
        pacman_position = gameState.getPacmanPosition()
        # Acciones legales de pacman
        legal_actions = gameState.getLegalPacmanActions()
        # Comprobando cuales de las acciones legales estan incluidas
        legal_N = 'North' in legal_actions
        legal_S = 'South' in legal_actions
        legal_E = 'East' in legal_actions
        legal_W = 'West' in legal_actions
        # Direcciones de pacman
        pacman_dir = gameState.data.agentStates[0].getDirection()
        # Agentes vivos
        living_ghosts = gameState.getLivingGhosts()
        # Posiciones de los fantasmas
        ghosts_positions = gameState.getGhostPositions()
        # Numero de agentes
        num_ghosts = gameState.getNumAgents() - 1

        # Direcciones de los fantasmas
        ghosts_directions = [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)]

        ghosts_distances = gameState.data.ghostDistances

        # Movimientos verticales
        mov_ver = 999999
        # Movimientos horizontales
        mov_hor = 999999
        # Movimientos a cada distancias
        min_distancia_norte = 999999
        min_distancia_sur = 999999
        min_distancia_oeste = 999999
        min_distancia_este = 999999
        # Variable spara guardar el número de fantasmas en cierta dirección
        num_ghost_N = 0
        num_ghost_S = 0
        num_ghost_W = 0
        num_ghost_E = 0
        # Variables para guardar las direcciones de los fantasmas más cercanos en cada dirección
        direccion_fantasma_N = 'None'
        direccion_fantasma_S = 'None'
        direccion_fantasma_W = 'None'
        direccion_fantasma_E = 'None'
        direccion_fantasma_cercano = "None"
        # Variable para la distancia al fantasma más cercano
        distancia_cercana_fantasma = 9999999
        # Variable para guardar los fantasmas no comidos
        num_living_ghosts = 0

        # numero de fantasmas mirando en cada direccion
        num_ghost_ori_N = 0
        num_ghost_ori_S = 0
        num_ghost_ori_E = 0
        num_ghost_ori_W = 0

        # Posiciones verticales para estar en la misma fila que el fantasma más cercano, en caso de empate se elige siempre al primero
        for agente in range(0, num_ghosts):
            if ghosts_distances[agente] is not None and (abs(ghosts_positions[agente][0] - pacman_position[0]) < mov_ver):
                mov_ver = abs(ghosts_positions[agente][0] - pacman_position[0])
            if ghosts_distances[agente] is not None and (abs(ghosts_positions[agente][1] - pacman_position[1]) < mov_hor):
                mov_hor = abs(ghosts_positions[agente][1] - pacman_position[1])
            # Oeste
            if ghosts_distances[agente] is not None and (ghosts_positions[agente][0] < pacman_position[0]):
                num_ghost_W += 1
                if min_distancia_oeste > abs(ghosts_positions[agente][0] - pacman_position[0]):
                    min_distancia_oeste = abs(
                        ghosts_positions[agente][0] - pacman_position[0])
                    direccion_fantasma_W = ghosts_directions[agente]

            # Este
            if ghosts_distances[agente] is not None and (ghosts_positions[agente][0] > pacman_position[0]):
                num_ghost_E += 1
                if min_distancia_este > abs(ghosts_positions[agente][0] - pacman_position[0]):
                    min_distancia_este = abs(
                        ghosts_positions[agente][0] - pacman_position[0])
                    direccion_fantasma_E = ghosts_directions[agente]

            # Norte
            if ghosts_distances[agente] is not None and (ghosts_positions[agente][1] > pacman_position[1]):
                num_ghost_N += 1
                if min_distancia_norte > abs(ghosts_positions[agente][1] - pacman_position[1]):
                    min_distancia_norte = abs(
                        ghosts_positions[agente][1] - pacman_position[1])
                    direccion_fantasma_N = ghosts_directions[agente]
            # Sur
            if ghosts_distances[agente] is not None and (ghosts_positions[agente][1] < pacman_position[1]):
                num_ghost_S += 1
                if min_distancia_sur > abs(ghosts_positions[agente][1] - pacman_position[1]):
                    min_distancia_sur = abs(
                        ghosts_positions[agente][1] - pacman_position[1])
                    direccion_fantasma_S = ghosts_directions[agente]

            # Calculando la dirección del fantasma más cercano
            if distancia_cercana_fantasma > (abs(ghosts_positions[agente][1]-pacman_position[1])+abs(ghosts_positions[agente][0]-pacman_position[0])):
                distancia_cercana_fantasma = abs(
                    ghosts_positions[agente][1]-pacman_position[1])+abs(ghosts_positions[agente][0]-pacman_position[0])
                direccion_fantasma_cercano = ghosts_directions[agente]

            if living_ghosts[agente+1]:
                num_living_ghosts = num_living_ghosts+1

            if ghosts_directions[agente] == "North":
                num_ghost_ori_N = num_ghost_ori_N+1
            if ghosts_directions[agente] == "West":
                num_ghost_ori_W = num_ghost_ori_W+1
            if ghosts_directions[agente] == "South":
                num_ghost_ori_S = num_ghost_ori_S+1
            if ghosts_directions[agente] == "East":
                num_ghost_ori_E = num_ghost_ori_E+1

        # En el caso de que no quede comida, la distancia más cercana se cambia de None a -1
        distance_food_nearest = -1
        if gameState.getDistanceNearestFood() is not None:
            distance_food_nearest = gameState.getDistanceNearestFood()

        map = gameState.getWalls()

        distancia_pared_norte = 0
        distancia_pared_sur = 0
        distancia_pared_este = 0
        distancia_pared_oeste = 0

        found = False
        for y in range(pacman_position[1], height):
            if not found and not map[pacman_position[0]][y]:
                distancia_pared_norte += 1
            elif not found:
                found = True

        found = False
        for y in range(pacman_position[1], 0, -1):
            if not found and not map[pacman_position[0]][y]:
                distancia_pared_sur += 1
            elif not found:
                found = True

        found = False
        for x in range(pacman_position[0], width):
            if not found and not map[x][pacman_position[1]]:
                distancia_pared_este += 1
            elif not found:
                found = True

        found = False
        for x in range(pacman_position[0], 0, -1):
            if not found and not map[x][pacman_position[1]]:
                distancia_pared_oeste += 1
            elif not found:
                found = True

        vision = [[0 for x in range(3)] for y in range(3)]

        for x in range(0, 3):
            for y in range(0, 3):
                if not(x == 1 and y == 1):
                    vision[x][y] = map[pacman_position[0] +
                                       x - 1][pacman_position[1] + y - 1]

                    if not vision[x][y]:
                        for agente in range(0, num_ghosts):
                            if ghosts_positions[agente][0] == pacman_position[0] + x - 1 and ghosts_positions[agente][1] == pacman_position[1] + y - 1:
                                vision[x][y] = 'Ghost'

        # Linea para escribir en fichero, se han casteado a string todos los valores para evitar errores en la escritura.

        lineData = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, ".format(
            str(self.countActions),
            str(width), str(height),
            str(pacman_position[0]), str(pacman_position[1]),
            str(legal_N), str(legal_S), str(legal_E), str(legal_W),
            str(pacman_dir),
            str(num_ghosts),
            str(num_living_ghosts),
            str(mov_ver),
            str(mov_hor),
            str(min_distancia_norte),
            str(min_distancia_sur),
            str(min_distancia_oeste),
            str(min_distancia_este),

            str(vision[0][0]),
            str(vision[0][1]),
            str(vision[0][2]),
            str(vision[1][0]),
            str(vision[1][2]),
            str(vision[2][2]),
            str(vision[2][1]),
            str(vision[2][2]),

            str(distancia_pared_norte),
            str(distancia_pared_sur),
            str(distancia_pared_este),
            str(distancia_pared_oeste),


            str(num_ghost_N),
            str(num_ghost_S),
            str(num_ghost_W),
            str(num_ghost_E),
            str(num_ghost_ori_N),
            str(num_ghost_ori_S),
            str(num_ghost_ori_W),
            str(num_ghost_ori_E),
            str(direccion_fantasma_N),
            str(direccion_fantasma_S),
            str(direccion_fantasma_W),
            str(direccion_fantasma_E),
            str(direccion_fantasma_cercano),
            str(distancia_cercana_fantasma),
            str(gameState.getNumFood()),
            str(distance_food_nearest),
            str(gameState.getScore())
        )

        return lineData


class QLearningAgent(BustersAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
    """

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.distancer = Distancer(gameState.data.layout)
        self.epsilon = 0.05
        self.alpha = 0.0
        self.discount = 0.95
        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3}
        self.get_actions = {0:"North", 1:"East", 2:"South", 3:"West"}
        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            # Number of rows depends on number of states
            state = self.transformState(gameState)
            self.initializeQtable(9216)

    def initializeQtable(self, nrows):
        self.q_table = np.zeros((nrows, len(self.actions)))

#     def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
#         "Initialize Q-values"
#         args['epsilon'] = epsilon
#         args['gamma'] = gamma
#         args['alpha'] = alpha
#         args['numTraining'] = numTraining
#         self.index = 0  # This is always Pacman
#         ReinforcementAgent.__init__(self, **args)

#         self.actions = {"north":0, "east":1, "south":2, "west":3, "exit":4}
#         self.table_file = open("qtable.txt", "r+")
# #        self.table_file_csv = open("qtable.csv", "r+")
#         self.q_table = self.readQtable()
#         self.epsilon = 0.05

    def chooseAction(self, gameState):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """

        action = self.selectAction(gameState)

        nextState = gameState.generateSuccessor(0, action)

        reward = self.getReward(gameState, action, nextState)
        self.update(gameState, action, nextState, reward)

        self.printInfo(nextState)

        return action

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()


    def computeDiscreteDistance(self,width,height, distance):
        border=math.floor((width*height)/3)
        a= math.floor(distance / border)
        if a>=3:
            a=2
        return a

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """
        '''
        return state[0](tantos estados como tamaño de row)+state[1](tantos estados como tamaño de columna)+(tantos estados como distancia máxima)*(length+width)
        '''

        transform_vision = { True: 0, False: 1, 'Ghost':2 , 'Food':3}
        transform_vertical = { "Aligned":0, "North":1 , "South":2 }
        transform_horizontal = { "Aligned":0, "East":1 , "West":2 }

        
        state = self.transformState(state)
        return transform_vertical[ state["closest_ghost_vertical"]] + 3 * transform_horizontal[ state["closest_ghost_horizontal"]] +  9 * transform_vision[state["vision_W"]] +  36 * transform_vision[state["vision_S"]] + 144 * transform_vision[state["vision_N"]] + 576 * transform_vision[state["vision_E"]]  + 2304 * state["wall_vertical"] + 4608 * state["wall_horizontal"]
        # return state["x_position"] + state["y_position"]*state["width"] + state["distance_to_ghost"]*(state["width"] + state["height"])
    def transformState(self, gameState):
        map = gameState.getWalls()
        food = gameState.data.food.data
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        pacman_position = gameState.getPacmanPosition()
        ghosts_positions = gameState.getGhostPositions()
        ghost_distances = gameState.data.ghostDistances
        gameState.getLivingGhosts()
        num_ghosts=gameState.getNumAgents() - 1
        vision = self.getPacmanVision(pacman_position, num_ghosts, ghosts_positions, map, gameState)

        # Get closest ghost and distance
        closest_distance, position_closest_ghost = self.getClosestGhost(ghost_distances,ghosts_positions)
        #Get discrete distance
        #discreteDistance=self.computeDiscreteDistance(width,height, closest_distance)
        x,y = self.calculateDirectionGhostPac(pacman_position, position_closest_ghost)
        
        wall_vertical = 0
        wall_horizontal = 0
        if y == 'South':
            for y_index in range(pacman_position[1], position_closest_ghost[1], -1):
                if map[pacman_position[0]][y_index]:
                    wall_vertical = 1
                    break

        else:
            for y_index in range(pacman_position[1], position_closest_ghost[1]):
                if map[pacman_position[0]][y_index]:
                    wall_vertical = 1
                    break

        if x == 'West':
            for x_index in range(pacman_position[0], position_closest_ghost[0], -1):
                if map[x_index][pacman_position[1]]:
                    wall_horizontal = 1
                    break
        else:
            for x_index in range(pacman_position[0], position_closest_ghost[0]):
                if map[x_index][pacman_position[1]]:
                    wall_horizontal = 1
                    break

        tupleState = {

            
            "wall_vertical": wall_vertical,
            "wall_horizontal": wall_horizontal,
            "closest_ghost_vertical": y,
            "closest_ghost_horizontal": x,
            # "vision_SW": vision[0][0],
            "vision_W": vision [0][1],
            # "vision_NW": vision[0][2],
            "vision_S": vision[1][0],        
            "vision_N": vision[1][2],
            #"vision_SE":  vision [2][0],     
            "vision_E": vision [2][1],
            #"vision_NE":  vision [2][2],
            "legal_actions": gameState.getLegalPacmanActions()
        }
        return tupleState

    def getPacmanVision(self,pacman_position,num_ghosts,ghost_positions, map, gameState):
        vision = [[0 for x in range(3)] for y in range(3)]
        for x in range(0, 3):
            for y in range(0, 3):
                if not(x == 1 and y == 1):
                    vision[x][y] = map[pacman_position[0] +
                                       x - 1][pacman_position[1] + y - 1]

                    if not vision[x][y]:
                        for agente in range(0, num_ghosts):
                            if ghost_positions[agente][0] == pacman_position[0] + x - 1 and ghost_positions[agente][1] == pacman_position[1] + y - 1:
                                vision[x][y] = 'Ghost'
                            elif gameState.hasFood(x,y):
                                vision[x][y] = 'Food'
        return vision
        
    
    def getClosestGhost(self, ghost_distances, ghosts_positions):
        """
        Returns the distance to the closest ghost and its positions
        """
        min = 99999999
        indice_min = 0
        for index in range(len(ghost_distances)):
            if ghost_distances[index] is not None and ghost_distances[index] <= min:
                min = ghost_distances[index]
                indice_min = index

        return min, ghosts_positions[indice_min]


    def calculateDirectionGhostPac(self,pacman_position, position_closest_ghost):
        x = 'Aligned'
        y = 'Aligned'
        if(position_closest_ghost[0]<pacman_position[0]):
            x="West"
        elif(position_closest_ghost[0]>pacman_position[0]):
            x="East"

        if(position_closest_ghost[1]<pacman_position[1]):
            y="South"
        elif(position_closest_ghost[1]>pacman_position[1]):
            y="North"
        return x,y
  
            
    
   

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]

        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalPacmanActions()
        if'Stop' in legalActions:
            legalActions.remove('Stop')
        if len(legalActions) == 0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if'Stop' in legalActions:
            legalActions.remove('Stop')
        if len(legalActions) == 0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in range(1,len(legalActions)):
            value = self.getQValue(state, legalActions[action])
            if value == best_value:
                best_actions.append(legalActions[action])
            if value > best_value:
                best_actions = [legalActions[action]]
                best_value = value

        return random.choice(best_actions)

    def selectAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalPacmanActions()
        action = None
        

        if len(legalActions) == 0:
            return action

        legalActions.remove("Stop")
        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        Good Terminal state -> reward 1
        Bad Terminal state -> reward -1
        Otherwise -> reward 0

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        # if reward == 1 or reward == -1:
        #     self.q_table[self.computePosition(state)][self.actions[action]] = (1-self.alpha) * self.q_table[self.computePosition(state)][self.actions[action]] + self.alpha * (reward + 0)
        # else:
       
        print(reward + self.discount*self.getValue(nextState))
        self.q_table[self.computePosition(state)][self.actions[action]] = (1-self.alpha) * self.q_table[self.computePosition(
            state)][self.actions[action]] + self.alpha * (reward + self.discount*self.getValue(nextState))

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextstate):
        # state = self.transformState(state)
        # nextstate = self.transformState(nextstate)
        # print(state["distance_to_ghost"])
        # if nextstate["distance_to_ghost"] is None:
        #     return 1
        # else:
        #     return 0
        score_now=state.getScore()
        score_later=nextstate.getScore()
        if(score_now<score_later):
            return abs(score_later-score_now)
        else:
            return 0
        
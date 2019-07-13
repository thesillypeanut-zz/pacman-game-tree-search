# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """


        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        walls = successorGameState.getWalls()
        ghostPositions = [ghostState.getPosition()
                          for ghostState in newGhostStates]
        newCapsules = successorGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        
        # clearly not the best evaluation fn. what's happening here?
        # TODO: revise this method
        if newPos in ghostPositions or (newPos[0]+1, newPos[1]) in ghostPositions or (newPos[0]-1, newPos[1]) in ghostPositions or (newPos[0], newPos[1]+1) in ghostPositions or (newPos[0], newPos[1]-1) in ghostPositions:
          return -score*2 if score > 0 else score*2

        x = newPos[0]
        y = newPos[1]

        if newPos in newCapsules:
          return score + 2

        # Look forward for pacman in the action direction until food, wall or a ghost is encountered.
        # If a food is encountered, add the inverse 1D distance to current score.
        # If a ghost is encountered and it is traveling towards the pacman, subtract the inverse dist.
        # If the ghost is scared, add the scared time to the score.
        # If a wall is encountered, subtract one from the score.
        # Note: If action is STOP, just subtract one from the score.
        if action == Directions.NORTH:
          while not walls[x][y] and not (x, y) in ghostPositions and not newFood[x][y]:
            y += 1

          manDist = manhattanDistance((newPos[0], newPos[1]), (x, y))

          if newFood[x][y]:
            score += 1/manDist if manDist > 0 else 1
          elif (x, y) in ghostPositions and newGhostStates[ghostPositions.index((x, y))].getDirection() == Directions.SOUTH:
            isGhostScared = newScaredTimes[ghostPositions.index((x, y))]
            score = score + 1/manDist if isGhostScared else score - 1/manDist
          else:
            score -= 1

        elif action == Directions.EAST:
          while not walls[x][y] and not (x, y) in ghostPositions and not newFood[x][y]:
            x += 1

          manDist = manhattanDistance((newPos[0], newPos[1]), (x, y))

          if newFood[x][y]:
            score += 1/manDist if manDist > 0 else 1
          elif (x, y) in ghostPositions and newGhostStates[ghostPositions.index((x, y))].getDirection() == Directions.WEST:
            isGhostScared = newScaredTimes[ghostPositions.index((x, y))]
            score = score + 1/manDist if isGhostScared else score - 1/manDist
          else:
            score -= 1

        elif action == Directions.SOUTH:
          while not walls[x][y] and not (x, y) in ghostPositions and not newFood[x][y]:
            y -= 1

          manDist = manhattanDistance((newPos[0], newPos[1]), (x, y))

          if newFood[x][y]:
            score += 1/manDist if manDist > 0 else 1
          elif (x, y) in ghostPositions and newGhostStates[ghostPositions.index((x, y))].getDirection() == Directions.NORTH:
            isGhostScared = newScaredTimes[ghostPositions.index((x, y))]
            score = score + 1/manDist if isGhostScared else score - 1/manDist
          else:
            score -= 1

        elif action == Directions.WEST:
          while not walls[x][y] and not (x, y) in ghostPositions and not newFood[x][y]:
            x -= 1

          manDist = manhattanDistance((newPos[0], newPos[1]), (x, y))

          if newFood[x][y]:
            score += 1/manDist if manDist > 0 else 1
          elif (x, y) in ghostPositions and newGhostStates[ghostPositions.index((x, y))].getDirection() == Directions.EAST:
            isGhostScared = newScaredTimes[ghostPositions.index((x, y))]
            score = score + 1/manDist if isGhostScared else score - 1/manDist
          else:
            score -= 1

        elif action == Directions.STOP:
          score -= 1

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def _DFMiniMax(gameState, agentIndex, currLevel):
          if currLevel == self.depth * numAgents or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

          legalMoves = gameState.getLegalActions(agentIndex)
          successors = [gameState.generateSuccessor(agentIndex, move) for move in legalMoves]
          if agentIndex == 0:
            # player is pacman, a max player
            return max([_DFMiniMax(successor, (agentIndex + 1) % numAgents, currLevel + 1) for successor in successors])
          else:
            # player is a ghost, a min player
            return min([_DFMiniMax(successor, (agentIndex + 1) % numAgents, currLevel + 1) for successor in successors])

        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, move) for move in legalMoves]
        scores = [_DFMiniMax(successor, 1, 1) for successor in successors]
        maxScore = max(scores)
        maxScoreIdx = scores.index(maxScore)

        return legalMoves[maxScoreIdx]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def _DFAlphaBeta(gameState, agentIndex, currLevel, alpha, beta):
          if currLevel == self.depth * numAgents or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

          legalMoves = gameState.getLegalActions(agentIndex)
          if agentIndex == 0:
            # player is pacman, a max player
            score = -float("inf")
            for move in legalMoves:
              score = max(score, _DFAlphaBeta(gameState.generateSuccessor(
                  agentIndex, move), (agentIndex + 1) % numAgents, currLevel + 1, alpha, beta))
              alpha = max(alpha, score)

              if currLevel == 0:
                scores.append(score)

              if beta <= alpha:
                break

            return score

          else:
            # player is a ghost, a min player
            score = float("inf")
            for move in legalMoves:
              score = min(score, _DFAlphaBeta(gameState.generateSuccessor(agentIndex, move), (agentIndex + 1) % numAgents, currLevel + 1, alpha, beta))
              beta = min(beta, score)

              if beta <= alpha:
                break

            return score

        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(0)
        scores = []
        _DFAlphaBeta(gameState, 0, 0, -float("inf"), float("inf"))
        maxScore = max(scores)
        maxScoreIdx = scores.index(maxScore)

        return legalMoves[maxScoreIdx]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def _DFExpectimax(gameState, agentIndex, currLevel):
          if currLevel == self.depth * numAgents or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

          legalMoves = gameState.getLegalActions(agentIndex)
          successors = [gameState.generateSuccessor(
              agentIndex, move) for move in legalMoves]
          if agentIndex == 0:
            # player is pacman, a max player
            return max([_DFExpectimax(successor, (agentIndex + 1) % numAgents,
              currLevel + 1) for successor in successors])
          else:
            # player is a ghost, a min player
            return float(sum([_DFExpectimax(successor, (agentIndex + 1) % numAgents,
              currLevel + 1) for successor in successors]))/len(successors)

        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(
            0, move) for move in legalMoves]
        scores = [_DFExpectimax(successor, 1, 1) for successor in successors]
        maxScore = max(scores)
        maxScoreIdx = scores.index(maxScore)

        return legalMoves[maxScoreIdx]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
        Returns a score that is a weighted sum of different features. The
        inverse manhattan distance is calculated from the current pacman position
        to each capsule, food and ghost. This calculated distance is then multiplied
        by a weight (found through trial and error) and added to or subtracted from the
        current score. The weighted distance for capsules, food and scared ghosts
        are added, while the weighted distance for non-scared ghosts are subtracted.

    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood().asList()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    currGhostPositions = [ghostState.getPosition() for ghostState in currGhostStates]
    currCapsules = currentGameState.getCapsules()

    score = currentGameState.getScore()
    weights = {
      "capsule": 10,
      "baseGhost": 6, # weight for ghosts is dynamically updated because of varying scared times
      "food": 2,
    }

    for capsule in currCapsules:
      manDist = manhattanDistance(currPos, capsule)
      capScore = weights["capsule"]/manDist if manDist > 0 else 0
      score += capScore

    for food in currFood:
      manDist = manhattanDistance(currPos, food)
      foodScore = weights["food"]/manDist if manDist > 0 else 0
      score += foodScore

    for i in range(len(currGhostStates)):
      ghostPos = currGhostPositions[i]
      manDist = manhattanDistance(currPos, ghostPos)
      scaredTime = currScaredTimes[i]
      weight = weights["baseGhost"] + scaredTime if scaredTime else weights["baseGhost"] + scaredTime
      ghostScore = weight/manDist if manDist > 0 else 0
      score = score + ghostScore if scaredTime else score - ghostScore

    return score

# Abbreviation
better = betterEvaluationFunction


from __future__ import annotations
from typing import List



class Node:
    def __init__(self, value=-1, depth=0, maxDepth=-1, subBranches:List[Branch] = []):
        self.value = value
        self.subBranches:List[Branch] = list(subBranches)
        self.depth = depth
        self.maxDepth = maxDepth
    def createSubBranch(self, probability:float, factor:float):
        newBranch = Branch(probability, factor, self)
        self.subBranches.append(newBranch)

    def spawnNodes(node:Node):
        if node.depth < node.maxDepth:
            for subBranch in node.subBranches:
                newNode = Node(node.value*subBranch.factor, node.depth+1, node.maxDepth)

                # newNode.createSubBranch(0.5, 1.1)
                # newNode.createSubBranch(0.5, 0.9)

                for newSubBranch in node.subBranches:
                    newNode.createSubBranch(newSubBranch.probability, newSubBranch.factor)
                subBranch.childNode = newNode
            for subBranch in node.subBranches:
                Node.spawnNodes(subBranch.childNode)
        elif node.depth == node.maxDepth:
            for subBranch in node.subBranches:
                newNode = Node(node.value*subBranch.factor, node.depth+1, node.maxDepth)
                subBranch.childNode = newNode

    def printTree(self):
        print(self, '\tchildren=', end='')
        if len(self.subBranches) > 0:
            childNodes = [branch.childNode for branch in self.subBranches]
            print([childNode.__str__() for childNode in childNodes])
            for childNode in childNodes:
                Node.printTree(childNode)
        else:
            print('')

    def __str__(self) -> str:
        return 'Node (value=' + str(round(self.value,2)) + ', depth=' + str(self.depth) + ')'
        

class Branch:
    def __init__(self, probabillity:float, factor:float, parentNode:Node, childNode = None) -> None:
        self.probability:float = probabillity
        self.factor:float = factor
        self.parentNode:Node = parentNode
        self.childNode:Node = childNode
    
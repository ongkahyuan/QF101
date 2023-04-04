from __future__ import annotations
from typing import List

from tree_model import Node, Branch

def main():
    firstNode = Node(value=10, depth=0, maxDepth=1)
    firstNode.createSubBranch(0.5, 1.1)
    firstNode.createSubBranch(0.5, 0.9)

    Node.spawnNodes(firstNode)
    Node.printTree(firstNode)
    

    pass

main()
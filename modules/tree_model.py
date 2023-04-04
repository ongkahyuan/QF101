from __future__ import annotations
from typing import List, Tuple
import math

class Node:
    def __init__(self, stock_value=-1, 
                    depth=0, probability_of_reaching=-1, parents:List[Node] = None, 
                    children:List[Node] = None):
        self.stock_value = stock_value
        self.euro_call_option_value = -1
        self.euro_put_option_value = -1
        self.amer_call_option_value = -1
        self.amer_put_option_value = -1
        self.parents:List[Node] = list(parents) if parents is not None else list()
        self.children:List[Node] = list(children) if children is not None else list()
        self.depth = depth
        self.probability_of_reaching:float = probability_of_reaching #represents probability of traversing to this node
    # Adds a child to the node and returns self
    def addChild(self, child:Node) -> Node:
        self.children.append(child)
        return self
    # Spawns (creates and adds) a child to the node and returns self
    def spawnChild(self, stock_value_factor:float, probability_of_traversing_to_child:float) -> Node:
        newNode = Node(self.stock_value*stock_value_factor, depth=self.depth+1,
                         probability_of_reaching=probability_of_traversing_to_child, parents=[self])
        self.addChild(newNode)
        return self


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

class BinomialTreeModel:
    def __init__(self, S_0, K, tau, r = 0.034, transition_probabilities:List[Tuple[float]] = None,
                     movement_factors:List[Tuple[float]] = None):
        """
        S_0:                        Stock price at time 0
        K:                          Strike price
        tau:                        Time to expiry in years
        r:                          Risk-free rate
        transition_probabilities:   List of (p, q)s; each tuple is used for one layer
        movement_factors:           List of (u, d)s; each tuple is used for one layer
        """
        # Given attributes
        self.S_0:float = S_0
        self.K:float = K
        self.tau:float = tau
        self.r:float = r
        self.transition_probabilities = list(transition_probabilities) if transition_probabilities is not None else list()
        self.movement_factors = list(movement_factors) if movement_factors is not None else list()

        if len(self.transition_probabilities) != len(self.movement_factors):
            raise ValueError('Length of list of transition probabilities and \
                                length of list of movement factors needs to be equal.')

        # Generated Attributes
        self.rootNode:Node = None
        self.nodes:List[List[Node]] = [[self.rootNode]] # Contains all nodes at layer i

        # Generated Compound Attributes
        self.N:int = 0
        self.discount_rate:float = 0
        self._regenerate_compound_attributes()

    def _regenerate_compound_attributes(self):
        self.N = len(self.transition_probabilities)
        if self.N != 0:
            self.discount_rate = self.discount_rate = math.exp(-self.r*(self.tau/self.N))

    # Generates the tree and returns self
    def generateTree(self) -> BinomialTreeModel:
        if self.N == 0:
            print("No transition probabilities and movement factors provided.")
            return self
        #print("Generating tree...")
        # Create the tree in a forward pass
        self.rootNode = Node(self.S_0, depth = 0)
        self.nodes = [[self.rootNode]]
        nodes_at_current_layer = [self.rootNode]
        for current_depth in range(self.N):
            p, q = self.transition_probabilities[current_depth]
            u, d = self.movement_factors[current_depth]
            newly_created_nodes = [] # nodes for the next layer
            for node in nodes_at_current_layer:
                node.spawnChild(u, p)
                node.spawnChild(d, q)
                newly_created_nodes.extend(node.children)
            nodes_at_current_layer = newly_created_nodes
            self.nodes.append(nodes_at_current_layer)

        # Calculate option prices of the last layer
        for terminal_node in self.nodes[-1]:
            u, d = self.movement_factors[-1]
            terminal_node.euro_call_option_value = max(0, terminal_node.stock_value-self.K)
            terminal_node.euro_put_option_value = max(0, self.K-terminal_node.stock_value)
            terminal_node.amer_call_option_value = max(0, terminal_node.stock_value-self.K)
            terminal_node.euro_put_option_value = max(0, self.K-terminal_node.stock_value)

        # Calculate the option prices from the back starting from the second last layer
        for current_depth in range(len(self.nodes)-2, -1, -1):
            current_layer = self.nodes[current_depth]
            for node in current_layer:
                # Calculate value of node from values of children nodes
                node.euro_call_option_value = sum([self.discount_rate * child.euro_call_option_value for child in node.children])
                node.euro_put_option_value = sum([self.discount_rate * child.euro_put_option_value for child in node.children])
                node.amer_call_option_value = sum([self.discount_rate * child.euro_put_option_value for child in node.children])
                node.amer_put_option_value = sum([self.discount_rate * child.euro_put_option_value for child in node.children])

                # Account for the fact that american options can be exercised anytime
                # American Call Option
                if (node.stock_value - self.K) > node.amer_call_option_value:
                    node.amer_call_option_value = node.stock_value - self.K
                # American Put Option
                if (self.K - node.stock_value) > node.amer_put_option_value:
                    node.amer_put_option_value = self.K - node.stock_value
        #print("Tree generated.")
        return self

    def generate_movement_and_transition_factors_method1(self, sigma, q=0.017, N=100) -> BinomialTreeModel:
        sigma = max(sigma, 0.001)
        u = math.exp(sigma*math.sqrt(self.tau/N))
        d = 1/u
        p = (math.exp((self.r-q)*(self.tau/N))-d)/(u-d)
        self.transition_probabilities = [[p,q]]*N
        self.movement_factors = [[u,d]]*N
        self._regenerate_compound_attributes()
        return self

    # Returns [(euro_call_opt_value, euro_put_opt_value), (amer_call_opt_value, amer_put_opt_value)]
    def getOptionValues(self) -> List[Tuple[float]]:
        return [
            (self.rootNode.euro_call_option_value, self.rootNode.euro_put_option_value),
            (self.rootNode.amer_call_option_value, self.rootNode.amer_put_option_value)
        ]
    @staticmethod
    def get_euro_option_values_method_1(S_0, K, tau, sigma, r = 0.034, N=10):
        model = BinomialTreeModel(S_0=S_0, K=K, tau=tau, r=r)
        model.generate_movement_and_transition_factors_method1(sigma=sigma, N=N)
        model.generateTree()
        euro_opt_values = model.getOptionValues()[0]
        return euro_opt_values
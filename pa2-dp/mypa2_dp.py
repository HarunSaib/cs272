from mymdp import MDP
import math

class ValueAgent:
    """Value-based Agent template (Used as a parent class for VIAgent and PIAgent)
    An agent should maintain:
    - q table (dict[state,dict[action,q-value]])
    - v table (dict[state,v-value])
    - policy table (dict[state,dict[action,probability]])
    - mdp (An MDP instance)
    - v_update_history (list of the v tables): [Grading purpose only] Every time when you update the v table, you need to append the v table to this list. (include the initial value)
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization

        Args:
            mdp (MDP): An MDP instance
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.            
        """        
        self.q = dict()
        self.v = dict()
        self.pi = dict()
        self.mdp = mdp
        self.thresh = conv_thresh
        self.v_update_history = list()

    def init_random_policy(self):
        """Initialize the policy function with equally distributed random probability.

        When n actions are available at state s, the probability of choosing an action should be 1/n.
        """        

        # Fetch and store all the states from mdp in 'states'
        states = self.mdp.states()

        # Initialize the state-value table V(s) to 0 for all the states fetched
        self.v = {s: 0.0 for s in states}
        # Then do the same for the Q-tables, and make some empty ones
        self.q = {s: {} for s in states}
        # Also initialize pi, which is gonna have the action % per state
        self.pi = {}

        # For each state, we need to give its actions a probabilty distribution (duh)
        for s in states:
            # Fetch the available actions from the current state 's' and save them
            actions = list(self.mdp.actions(s))

            # If there arent any actions, then it's a terminal state so we done
            if not actions:
                continue
            
            # Otherwise, calculate the "uniform probabilty", which is just 1 / num of actions (ezpz)
            prob = 1.0 / float(len(actions))

            # Then just give that probabilty to each of the current state 's's actions (english is hard)
            self.pi[s] = {a: prob for a in actions}
        
        # And now we're done with initializing the random policy (yipeee)

    def computeq_fromv(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Given a state-value table, compute the action-state values.
        For deterministic actions, q(s,a) = E[r] + v(s'). Check the lecture slides.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            dict[str,dict[str,float]]: a q value table {state:{action:q-value}}
        """
        
        # Make the q-table for the state, holds the available actions, and that action's expected return (not reward)
        q = {}

        # Then for each state in the mdp (again)
        for s in self.mdp.states():
            # Make a temp var to store the current state 's's q-table
            q_s = {}

            # And then for each action of that state
            for a in self.mdp.actions(s):
                # Make a 'total' to predict the estimated/expected (and discounted) 'return' from taking action 'a'
                total = 0.0

                # And THEN (again) iterate over all the possible successor states & their probs
                # self.mdp.T(s, a) gets the 'T' (transition model) or chance you end up in s' after taking action 'a' from s
                for (s_prime, prob) in self.mdp.T(s, a):
                    # If that (current) successor has no chance of being reached, don't bother with it
                    if prob == 0.0:
                        continue

                    # Fetch n store the immediate reward in 'r' that we would get by getting to s' (s_prime)
                    r = self.mdp.R(s, a, s_prime)

                    # update the total based on the expected (discounted) contribution from this successor state
                    # p * (r + gamma * v(s'))
                    total += prob + (r + self.mdp.gamma * v[s_prime])

                # Now that we looped through all of action a's successors, update a's q-table
                q_s[a] = total

            # Now that we looped through all of the actions in s, save its action-values
            q[s] = q_s

        # Now that we computed the q, we're done :D (yipeee again)
        return q
    
    def greedy_policy_improvement(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Greedy policy improvement algorithm. Given a state-value table, update the policy pi.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        
        # First things first, get the action-values q(s, a) from the state-values 'v'
        q = self.computeq_fromv(v)

        # Start making the new pi policy
        new_pi = {}

        # Once again, for each state s in state, find the best action in that state (greedy)
        for s in self.mdp.states():
            # Save the actions available in s in 'actions'
            actions = list(self.mdp.actions(s))

            # If no actions are available, then it's a terminal state so don't bother
            if not actions:
                continue

            # Try to find the "argMax" of q (best action) and save it
            max_q = max((q[s][a] for a in actions), default = None)

            # Now save all the bestest of actions where the q(s, a) == max_q 
            # (with an edge case tolerance range 'epsilon')
            bestest = []

            # speaks for itself
            for a in actions:
                # If the difference is miniscule cuz floats are weird
                if abs(q[s][a] - max_q) < 1e-12:
                    # Then we save that greedy action into bestest
                    bestest.append(a)

            # Now distribute the probability uniformly for the bestest actions
            prob = 1.0 / float(len(bestest))
            # Make the row to hold the probs of all the actions {action : prob}
            row = {}
            for a in actions:
                # then if the action is in bestest, save that action and it's new prob in the row
                row[a] = prob if a in bestest else 0.0
                    
            # And now update the pi so it has the new (greedy) row for this current state s
            new_pi[s] = row

        # Now that we've done the 'greedy improvement', update the q-table, and policy to reflect that
        self.q = q
        self.pi = new_pi

        # And then return the new policy and we're done (yipeee again)
        return new_pi


    # TODO
    def check_term(self, v: dict[str,float], next_v: dict[str,float]) -> bool:
        """Return True if the state value has NOT converged.
        Convergence here is defined as follows: 
        For ANY state s, the update delta, abs(v'(s) - v(s)), is within the threshold (self.thresh).

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}
            next_v (dict[str,float]): a state value table (after update)

        Returns:
            bool: True if continue; False if converged
        """
        pass     


class PIAgent(ValueAgent):
    """Policy Iteration Agent class
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    # TODO
    def __iter_policy_eval(self, pi: dict[str,dict[str,float]]) -> dict[str,float]:
        """Iterative policy evaluation algorithm. Given a policy pi, evaluate the value of states (v).

        This function should be called in policy_iteration().

        Args:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}

        Returns:
            dict[str,float]: state-value table {state:v-value}
        """
        pass

    # TODO
    def policy_iteration(self) -> dict[str,dict[str,float]]:
        """Policy iteration algorithm. Iterating iter_policy_eval and greedy_policy_improvement, update the policy pi until convergence of the state-value function.

        You must use:
         - __iter_policy_eval
         - greedy_policy_improvement        

        This function is called to run PI. 
        e.g.
        mdp = MDP("./mdp1.json")
        dpa = PIAgent(mdp)
        dpa.policy_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        pass


class VIAgent(ValueAgent):
    """Value Iteration Agent class
    """
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy     

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    # TODO
    def value_iteration(self) -> dict[str,dict[str,float]]:
        """Value iteration algorithm. Compute the optimal v values using the value iteration. After that, generate the corresponding optimal policy pi.

        You must use:
         - greedy_policy_improvement           

        This function is called to run VI. 
        e.g.
        mdp = MDP("./mdp1.json")
        via = VIAgent(mdp)
        via.value_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        pass

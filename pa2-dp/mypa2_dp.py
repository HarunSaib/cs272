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

        # For all the known states in the mdp
        for s in self.mdp.states():
            # Calculate the difference (delta) between the current and next state's values
            delta = abs(next_v[s] - v[s])

            # If the difference is big enough (greater than the threshhold)
            if delta > self.thresh:
                # We're not done, so it's a go to continue iterating
                return True
        
        # Otherwise, if all the states are converged enough, we're done (yayyyy we did it)
        return False


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

    def __iter_policy_eval(self, pi: dict[str,dict[str,float]]) -> dict[str,float]:
        """Iterative policy evaluation algorithm. Given a policy pi, evaluate the value of states (v).

        This function should be called in policy_iteration().

        Args:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}

        Returns:
            dict[str,float]: state-value table {state:v-value}
        """
        
        # this is gonna be ugly :c

        # Just get states cuz we're gonna be reusing it
        states = self.mdp.states()

        # get the (shallow copy of the) agent's current v (values of states)
        v = {s: self.v.get(s, 0.0) for s in states}

        # it's 1am I dont wanna write comments for this
        # too bad.
        # Quickly take a 'snapshot' of the current v to keep track of the history of them
        self.v_update_history.append({s: v[s] for s in states})

        # Now, start doing the Bellman expectation equation, and stop when it converges
        while True: # ew
            # Start storing the next expected v's calculated by bellman
            next_v = {}

            # yk what this means by now
            for s in states:
                # Make a default val if the state has no actions under this policy (yet)
                v_s = 0.0

                # If the state has actions to choose from
                if s in pi: # since terminal states aren't in pi
                    # Then for all the actions available in the state
                    for a in self.mdp.actions(s):
                        # Then get the weighted action-values of the state s, following pi (defaulting to 0)
                        pi_sa = pi[s].get(a, 0.0)
                        # If they don't contribute at all, dont bother with them
                        if pi_sa == 0.0:
                            continue
                        
                        # Make an expected return to calculate taking action a and then continuing with v
                        total = 0.0

                        # too sleepy to comment this I've said it before
                        for (s_prime, prob) in self.mdp.T(s, a):
                            # If we never choose the action, skip it
                            if prob == 0.0:
                                continue

                            # get the immidiate reward (again) of if we take action a from s to s'
                            r = self.mdp.R(s, a, s_prime)
                            # update the (discounted) total again (yayy hes back)
                            total += prob * (r + self.mdp.gamma * v[s_prime])

                        # Get the weight according to the policy's prob of taking action a in s
                        v_s += pi_sa * total
                
                # Save the new state value for this state
                next_v[s] = v_s

            # Take another snapshot after the iteration for comparison (next_v instead of v this time)
            self.v_update_history.append({s: next_v[s] for s in states})

            # If we haven't converged yet (why isn't that function named converged)
            if not self.check_term(v, next_v):
                # if we converged, we wanna use the new converged values so update v to that
                v = next_v
                break # FREEDOM FROM THE WHILE LOOP

            # Otherwise we're not done yet :c so update v and move onto the next iteration
            v = next_v

        # once we're free of the loop, save the bestest newest v in the agent
        self.v = v

        # and then return it for policy_iteration to use cuz we're so polite n helpful
        return v

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
        
        # This is where the ✨magic✨ happens, alternating between the policy evaluation and improvement until perfection

        # Keep alternating and iterating until minimal change
        while True:
            # First evaluate the current policy pi, and save that returned v
            v = self.__iter_policy_eval(self.pi)
            
            # Keep track of the old pi for comparison
            old_pi = {}
            # Iterate through all the states
            for s in self.mdp.states():
                if s in self.pi: # if the state isn't terminal (cuz we dont save those)
                    old_pi[s] = dict(self.pi[s]) # copy the pi for that state into old_pi
            
            # now that we saved the old pi, time to greed a new one
            new_pi = self.greedy_policy_improvement(v)

            # now we compare ALL of the old & new pi's (with range of tolerance again)
            stable = True
            for s in self.mdp.states():
                actions = list(self.mdp.actions(s)) # get the actions
                if not actions: # if terminal, skip
                    continue
                
                # Try and get the old and new pi rows for the state s, defaulting to empty
                old_row = old_pi.get(s, {})
                new_row = new_pi.get(s, {})

                # Then check for each action
                for a in actions:
                    # get the probability of each action in the row, def to 0
                    # aka the probability of taking action a under the old & new policies
                    old_prob = old_row.get(a, 0.0)
                    new_prob = new_row.get(a, 0.0)

                    # check if the difference/change is too big to stop (with epsilon forgiveness cuz we nice)
                    if abs(new_prob - old_prob) > 1e-12:
                        stable = False # not converged enough yet
                        break # outta the for loop

                if not stable:
                    break # outta the other for loop
            
            # If ALL of the probs of ALL of the actions barely changed, then it's converged enough
            # and so it's optimal enough for us to stop
            if stable:
                break

        # Finally, return the bestest of the bestest policy
        return self.pi

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

        # some of the code is basically the same as policy eval so just copy that over to edit

        # This is the same
        states = self.mdp.states()
        v = {s: self.v.get(s, 0.0) for s in states}
        self.v_update_history.append({s: v[s] for s in states})

        # Still the same
        while True:
            next_v = {}

            # Now we start changing
            for s in states:
                # make a list of the available actions in s
                actions = list(self.mdp.actions(s))

                # If it's a terminal state with no actions, skip it
                if not actions: 
                    next_v[s] = 0.0 # (set it to 0 first cuz we're not messy)
                    continue

                # now we gotta find the argmax again
                bestest = None
                for a in actions:
                    total = 0.0 # expected return again
                    # this thing again I've explained it before
                    for (s_prime, prob) in self.mdp.T(s, a):
                        if prob == 0.0: # if the agent doesn't care about a, we don't either
                            continue

                        # get the reward same as last time
                        r = self.mdp.R(s, a, s_prime)
                        # update the (discounted) total again
                        total += prob * (r + self.mdp.gamma * v[s_prime])

                    # look for the bestest argmax 
                    if bestest is None or (total > bestest):
                        bestest = total
                

                # Save the new value for this state, same logic, different numbers
                next_v[s] = bestest if bestest is not None else 0.0

            # Take another snapshot same as last time
            self.v_update_history.append({s: next_v[s] for s in states})

            # Same as last time
            if not self.check_term(v, next_v):
                v = next_v
                break

            v = next_v

        # save the bestest v to the agent for future use again
        self.v = v

        # now that we have our v, we use it to derive the optimal pi from our greedy helper
        pi = self.greedy_policy_improvement(self.v)
        # and then return our bestest and greediest policy (mwhahahaha)
        return pi

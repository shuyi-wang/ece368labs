import numpy as np
import graphics
import rover


def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps
    for i in range(0, num_time_steps):
        marginals[i] = rover.Distribution()
        forward_messages = rover.Distribution()
        backward_messages = rover.Distribution()

    # TODO: Compute the forward messages
    # Initialize pi_0 with prior_distribution
    pi_0 = prior_distribution

    # Update forward messages at time step 0
    for i in pi_0:
        z0_obs = observation_model(i)
        for j in z0_obs:
            if observations[0] is None:
                forward_messages[0][i] = pi_0[i] * 1
            elif j == observations[0]:
                forward_messages[0][i] = pi_0[i] * z0_obs[j]

    # Renormalize forward messages at time step 0
    forward_messages[0].renormalize()

# Recursion for time steps 1 to num_time_steps
    for n in range(1, num_time_steps):
        A = dict()  # Accumulate intermediate results

    # Update A for time step n-1
    for i in forward_messages[n-1]:
        temp = transition_model(i)
        for j in temp:
            if j in A.keys():
                A[j] += temp[j] * forward_messages[n-1][i]
            else:
                A[j] = temp[j] * forward_messages[n-1][i]

    # Update forward messages at time step n
    for i in A:  # all the z_i's
        zi_obs = observation_model(i)
        for j in zi_obs:
            if observations[n] is None:
                forward_messages[n][i] = A[i] * 1
            elif j == observations[n]:
                forward_messages[n][i] = A[i] * zi_obs[j]

    # Renormalize forward messages at time step n
    forward_messages[n].renormalize()

    # TODO: Compute the backward messages
    for i in all_possible_hidden_states:
        backward_messages[num_time_steps-1][i] = 1

    for n in range(num_time_steps - 2, -1, -1):
        A = dict()

        for i in backward_messages[n + 1]:
            zi_obs = observation_model(i)  # p((x_n, y_n)|z_n)
            for j in zi_obs:
                if (observations[n + 1] == None):
                    A[i] = backward_messages[n + 1][i] * 1

                elif (j == observations[n + 1]):
                    A[i] = backward_messages[n + 1][i] * zi_obs[j]

        for i in A:
            for all_h in all_possible_hidden_states:
                temp_transition = transition_model(all_h)
                for k in temp_transition:
                    if (k == i and all_h in backward_messages[n].keys()):
                        backward_messages[n][all_h] += temp_transition[k] * A[i]
                    elif (k == i and all_h not in backward_messages[n].keys()):
                        backward_messages[n][all_h] = temp_transition[k] * A[i]
        backward_messages[n].renormalize()

    # TODO: Compute the marginals
    for i in range(0, num_time_steps):
        sum = 0
        for fm in forward_messages[i]:
            if forward_messages[i][fm]*backward_messages[i][fm] != 0:
                marginals[i][fm] = forward_messages[i][fm] * \
                    backward_messages[i][fm]
                sum = sum + marginals[i][fm]
        for fm in marginals[i].keys():
            marginals[i][fm] = marginals[i][fm]/sum
    return marginals


def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps
    z_previous = [None] * num_time_steps

    # Initialization
    w[0] = rover.Distribution({})
    initial_observed_position = observations[0]
    for z0 in all_possible_hidden_states:
        initial_prob_position_on_state = 1 if initial_observed_position is None else observation_model(z0)[
            initial_observed_position]
        prior_z0 = prior_distribution[z0]
        if initial_prob_position_on_state != 0 and prior_z0 != 0:
            w[0][z0] = np.log(
                initial_prob_position_on_state) + np.log(prior_z0)

    # When i >= 1
    for i in range(1, num_time_steps):
        w[i] = rover.Distribution({})
        z_previous[i] = dict()
        observed_position = observations[i]
        for zi in all_possible_hidden_states:
            prob_position_on_state = 1 if observed_position is None else observation_model(zi)[
                observed_position]
            max_term = -np.inf
            for zi_minus_1 in w[i-1]:
                if transition_model(zi_minus_1)[zi] != 0:
                    potential_max_term = np.log(transition_model(
                        zi_minus_1)[zi]) + w[i-1][zi_minus_1]
                    if potential_max_term > max_term and prob_position_on_state != 0:
                        max_term = potential_max_term
                        z_previous[i][zi] = zi_minus_1
            if prob_position_on_state != 0:
                w[i][zi] = np.log(prob_position_on_state) + max_term

    # Backtrack to find z0 to zn
    # First, find zn* (the last)
    max_w = -np.inf
    for zi in w[num_time_steps-1]:
        potential_max_w = w[num_time_steps-1][zi]
        if potential_max_w > max_w:
            max_w = potential_max_w
            estimated_hidden_states[num_time_steps-1] = zi

    for i in range(1, num_time_steps):
        estimated_hidden_states[num_time_steps-1-i] = z_previous[num_time_steps -
                                                                 i][estimated_hidden_states[num_time_steps-i]]

    return estimated_hidden_states


if __name__ == '__main__':

    enable_graphics = True

    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'

    # load data
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution = rover.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(),
          key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

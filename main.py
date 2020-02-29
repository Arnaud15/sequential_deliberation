import numpy as np
import cvxpy as cp


def social_cost(agents, outcome):
    """
    just a sum
    """
    cost = 0.
    for agent_ix in range(agents.shape[0]):
        cost += np.abs(agents[agent_ix] - outcome).sum()
    return cost


def compute_nash(agents, disagreement_outcome):
    """
    convex opt
    """
    assert agents.shape[0] == 2
    o = cp.Variable(agents.shape[1])
    constraints = []
    constraints.append(cp.sum(o) == 1)
    constraints.append(o >= 0)
    distances = [np.abs(agents[i] - disagreement_outcome).sum() for i in range(2)]
    objective = cp.sum([cp.log(distances[i] - cp.norm1(agents[i] - o)) for i in range(2)])
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()
    if problem.status == "infeasible":
        return disagreement_outcome
    else:
        return o.value


def pick_agents(agents, num):
    """
    random choice
    """
    num_agents = agents.shape[0]
    indexes_picked = np.random.choice(num_agents, size=num, replace=False)
    return agents[indexes_picked]


def generate_random_agents(num_agents, dimension):
    """
    simplex with exponentials
    param: num_agents int
    param: dimension int
    return: agents = 2D array of shape (num_agents, dimension) on the dimension-dimensional simplex
    """
    agents = np.random.rand(num_agents, dimension)
    agents = - np.log(agents)
    agents = agents / np.expand_dims(agents.sum(1), -1)
    return agents


def optimal_cost(agents):
    o = cp.Variable(agents.shape[1])
    constraints = []
    constraints.append(cp.sum(o) == 1)
    constraints.append(o >= 0)
    objective = cp.sum([cp.norm1(agents[i] - o) for i in range(agents.shape[0])])
    problem = cp.Problem(cp.Minimize(objective), constraints)
    optimal_c = problem.solve()
    if problem.status == "infeasible":
        raise ValueError("Problem no feasible")
    return optimal_c


def sequential_deliberation(agents, num_expectation_samples, num_mc_steps):
    expected_cost = 0.
    for _ in range(num_expectation_samples):
        a = pick_agents(agents, 1)
        # print(f"starting from point: {a}")
        for _ in range(num_mc_steps):
            agents_picked = pick_agents(agents, 2)
            a = compute_nash(agents_picked, disagreement_outcome=a)
            # print(f"new point is: {a}")
        # print(f"final point is {a}")
        expected_cost += social_cost(agents, a)
    expected_cost /= num_expectation_samples
    return expected_cost


def experiment(n_agents, dimension, n_simulations, num_samples, num_steps):
    distortions = np.zeros(n_simulations)
    for simulation_ix in range(n_simulations):
        agents = generate_random_agents(n_agents, dimension)
        algo_cost = sequential_deliberation(agents=agents,
                                            num_expectation_samples=num_samples,
                                            num_mc_steps=num_steps)
        print(f"algo cost is {algo_cost}")
        cost_star = optimal_cost(agents)
        print(f"optimal cost is {cost_star}")
        distortions[simulation_ix] = algo_cost / cost_star
    return distortions
                

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nagents", type=int, default=10)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--nsim", type=int, default=10)
    parser.add_argument("--nsamples", type=int, default=10)
    parser.add_argument("--nsteps", type=int, default=15)
    args = parser.parse_args()
    distortions = experiment(args.nagents, args.dim, args.nsim, args.nsamples, args.nsteps)
    print(distortions.mean(), distortions.std(), distortions.max())
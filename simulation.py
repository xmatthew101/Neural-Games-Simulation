import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Set seed for reproducibility
np.random.seed(15)


#helper functions

#computes joint distribution p(s, y1, ..., yN)
def joint_distribution(N, M, p_s, strategies):
    joint_distr = {}
    for s in range(M):
        ps = p_s[s]
        #iterates over all 2^N combinations of messages from the N neurons
        for m in range(2 ** N):
            msg = [(m >> i) & 1 for i in range(N)] #converts m into binary
            p_msg = ps
            #multiplies all yjk(s)
            for k in range(N):
                p_msg *= strategies[k, s, msg[k]]
            joint_distr[(s, tuple(msg))] = p_msg
    return joint_distr

#computes marginal distribution of message combinations, p(y1, ..., yN) by summing over possible stimuli
#takes joint_distribution as an input
def message_distribution(joint_distr):
    msg_distr = {}
    # marginal over s
    for (s, msg), p_msg in joint_distr.items():
        msg_distr[msg] = msg_distr.get(msg, 0) + p_msg
    return msg_distr

#computes conditional entropy H(S|Y1, ..., YN)
def conditional_entropy(N, M, joint_distr, marginal_distr):
    H = 0
    for msg in marginal_distr:
        p_yk = marginal_distr[msg]
        if p_yk == 0:
            continue
        #computes p(s | msg)
        ps_given_msg = []
        for s in range(M):
            num = joint_distr.get((s, msg), 0)
            if p_yk > 0:
                ps_given_msg.append(num / p_yk)
            else:
                ps_given_msg.append(0)
        #subfunction to calculate entropy
        def marginal_entropy(probs):
            return -sum(p * np.log2(p) for p in probs if p > 0)
        H -= p_yk * marginal_entropy(ps_given_msg)
    return H

#computes marginal distributions p(yk=0) and p(yk=1) for each neuron k
#also computes p(yk=a, yj=b) for k!=j
def compute_joint_marginals(N, M, joint_distr):
    marginals = {k: {0: 0, 1: 0} for k in range(N)}
    joint_pairwise = {}
    for (s, msg), p in joint_distr.items():
        for k in range(N):
            marginals[k][msg[k]] += p
            for j in range(N):
                if j == k: continue
                key = (k, j, msg[k], msg[j])
                joint_pairwise[key] = joint_pairwise.get(key, 0) + p
    return marginals, joint_pairwise

def compute_EV(N, M, k, strategies, mu, p_s, neuron_types=[], kappa=-1.0):
    """
    Compute expected utility EV_k(Yk; Y_-k) as in eq (8)
    EV_k = sum_{j != k} H(Yk | Yj) - mu H(Yk | S)
    """
    joint_distr = joint_distribution(N, M, p_s, strategies)
    msg_distr = message_distribution(joint_distr)
    #compute H(Yk|Yj)
    cond_ent = 0
    if kappa < 0:
        kappa = 0.5
    for j in range(N):
        if j == k:
            continue
        #joint distribution of (Yk, Yj)
        p_joint = { (a,b):0 for a in [0,1] for b in [0,1] }
        for (s,msg), prob in joint_distr.items():
            p_joint[(msg[k], msg[j])] += prob
        p_yj = {b: p_joint[(0,b)] + p_joint[(1,b)] for b in [0,1]}
        for b in [0,1]:
            if p_yj[b] > 0:
                for a in [0,1]:
                    p_cond = p_joint[(a,b)] / p_yj[b] if p_yj[b] > 0 else 0
                    if p_cond > 0:
                        if len(neuron_types) == 0:
                            cond_ent -= p_joint[(a,b)] * np.log2(p_cond)
                        elif len(neuron_types) == N:
                            neuron_k_type = neuron_types[k]
                            neuron_j_type = neuron_types[j]
                            #excitatory neurons
                            if neuron_j_type == "E":
                                if neuron_k_type == "I":
                                    cond_ent -= p_joint[(a,b)] * np.log2(p_cond)
                                else:
                                    cond_ent -= 2*(1-mu) * p_joint[(a, b)] * np.log2(p_cond)
                            #inhibitory neurons
                            elif neuron_j_type == "I":
                                if neuron_k_type == "E":
                                    cond_ent += p_joint[(a,b)] * np.log2(p_cond)
                                else:
                                    cond_ent += 2*(1-kappa) * p_joint[(a,b)] * np.log2(p_cond)
                            else:
                                print("Neuron types do not map correctly")
                                return
                        else:
                            print("Neuron types do not map correctly")
                            return
    #compute H(Yk|S)
    H_Yk_S = 0
    for s in range(M):
        for a in [0,1]:
            p = strategies[k, s, a]
            if p > 0:
                H_Yk_S -= p_s[s] * p * np.log2(p)
    if len(neuron_types) == 0:
        EV = cond_ent - mu * H_Yk_S
    elif len(neuron_types) == N:
        neuron_k_type = neuron_types[k]
        if mu < 0 or mu > 1:
            mu = 0.5
        # excitatory neurons
        if neuron_k_type == "E":
            EV = cond_ent - 2*mu * H_Yk_S
        # inhibitory neurons
        elif neuron_k_type == "I":
            EV = cond_ent + 2*kappa * H_Yk_S
        else:
            print("Neuron types do not map correctly")
            return
    else:
        print("Neuron types do not map correctly")
        return
    return EV

#central finite-difference gradient descent used as numerical method
def optimize_neuron_numerical(N, M, k, strategies, mu, p_s, lr, neuron_types=[], kappa=-1.0):
    grad = np.zeros(M)
    eps = 1e-12
    for i in range(M):
        original = strategies[k, i, 1]
        #central difference approximation
        strategies[k, i, 1] = np.clip(original + eps, 0, 1)
        strategies[k, i, 0] = 1 - strategies[k, i, 1]
        # joint_distr_plus = joint_distribution(N, M, p_s, strategies)
        # msg_distr_plus = message_distribution(joint_distr_plus)
        # H_plus = conditional_entropy(N, M, joint_distr_plus, msg_distr_plus)
        EV_plus = compute_EV(N, M, k, strategies, mu, p_s, neuron_types, kappa)

        strategies[k, i, 1] = np.clip(original - eps, 0, 1)
        strategies[k, i, 0] = 1 - strategies[k, i, 1]
        # joint_distr_minus = joint_distribution(N, M, p_s, strategies)
        # msg_distr_minus = message_distribution(joint_distr_minus)
        # H_minus = conditional_entropy(N, M, joint_distr_minus, msg_distr_minus)
        EV_minus = compute_EV(N, M, k, strategies, mu, p_s, neuron_types, kappa)

        # grad[i] = (H_plus - H_minus) / (2 * eps)
        grad[i] = (EV_plus - EV_minus) / (2 * eps)
        strategies[k, i, 1] = original  # reset
        strategies[k, i, 0] = 1 - original
    #gradient descent
    strategies[k, :, 1] += lr * grad
    strategies[k, :, 1] = np.clip(strategies[k, :, 1], 0, 1)
    strategies[k, :, 0] = 1 - strategies[k, :, 1]
    return strategies

def optimize_neuron_benchmark(N, M, k, strategies, mu, p_s, lr):
    grad = np.zeros(M)
    eps = 1e-12
    for i in range(M):
        original = strategies[k, i, 1]
        #central difference approximation
        strategies[k, i, 1] = np.clip(original + eps, 0, 1)
        strategies[k, i, 0] = 1 - strategies[k, i, 1]
        joint_distr_plus = joint_distribution(N, M, p_s, strategies)
        msg_distr_plus = message_distribution(joint_distr_plus)
        H_plus = conditional_entropy(N, M, joint_distr_plus, msg_distr_plus)

        strategies[k, i, 1] = np.clip(original - eps, 0, 1)
        strategies[k, i, 0] = 1 - strategies[k, i, 1]
        joint_distr_minus = joint_distribution(N, M, p_s, strategies)
        msg_distr_minus = message_distribution(joint_distr_minus)
        H_minus = conditional_entropy(N, M, joint_distr_minus, msg_distr_minus)

        grad[i] = (H_plus - H_minus) / (2 * eps)

        strategies[k, i, 1] = original  # reset
        strategies[k, i, 0] = 1 - original
    #gradient descent
    strategies[k, :, 1] += lr * grad
    strategies[k, :, 1] = np.clip(strategies[k, :, 1], 0, 1)
    strategies[k, :, 0] = 1 - strategies[k, :, 1]
    return strategies

def optimize_neuron_random(N, M, k, strategies, mu, p_s, neuron_types=[]):
    #Random jump optimization, trying a random strategy for neuron k
    current_EV = compute_EV(N, M, k, strategies, mu, p_s, neuron_types)
    #generates first column of random strategy
    random = np.random.rand(M, 1)
    #pieces random and 1-random columns together for new random strategy
    new_strategy = np.hstack((random, np.full((M, 1), 1) - random))
    #backup old strategy for neuron k
    old_strategy = strategies[k].copy()
    #test new strategy
    strategies[k] = new_strategy
    new_EV = compute_EV(N, M, k, strategies, mu, p_s, neuron_types)
    if new_EV < current_EV:
        #revert
        strategies[k] = old_strategy
    return strategies

#analytical gradient from Section 6
def optimize_neuron_analytical(N, M, k, strategies, mu, p_s, lr):
    grad = np.zeros(M)
    eps = 1e-12
    joint_distr = joint_distribution(N, M, p_s, strategies)
    _, joint_pairwise = compute_joint_marginals(N, M, joint_distr)

    for i in range(M):
        yk0 = strategies[k, i, 0]
        yk1 = strategies[k, i, 1]
        dEV = 0

        for j in range(N):
            if j == k:
                continue
            yj0 = strategies[j, i, 0]
            yj1 = strategies[j, i, 1]
            ps = p_s[i]

            #terms from Section 6 (log ratio of joint probs)
            a = 0
            b = 0
            c = 0
            d = 0
            p00 = ps * yj0 * yk0 + a
            p10 = ps * yj1 * yk0 + b
            p01 = ps * yj0 * (1-yj1) + c
            p11 = ps * yj1 * (1-yk0) + d

            #add epsilon to avoid log(0)
            dEV -= ps * (
                yj0 * np.log2((p00 + eps) / (p01 + eps)) +
                yj1 * np.log2((p10 + eps) / (p11 + eps))
            )

        #entropy regularization (self entropy), p_s in 13 but not on page 9?
        dEV += mu * p_s[i] * np.log2((yk0 + eps) / (yk1 + eps))

        grad[i] = dEV

    #gradient ascent step
    strategies[k, :, 1] += lr * grad
    strategies[k, :, 1] = np.clip(strategies[k, :, 1], 0, 1)
    strategies[k, :, 0] = 1 - strategies[k, :, 1]
    return strategies


#main simulation function
def simulation(N, M, T, mu, lr, signal_distr, method, neuron_types=[], kappa=-1.0):
    #each neuron k has a strategy matrix Y_k of shape (M, 2)
    #y_k[i, 1] = Pr(Y_k = 1 | s = s_i)
    #y_k[i, 0] = 1 - y_k[i, 1] = Pr(Y_k = 0 | s = s_i)
    #initialize random strategies
    strategies = np.random.rand(N, M)
    strategies = np.stack([1 - strategies, strategies], axis=2)  # shape (N, M, 2)
    strategy_traj = np.zeros((T, N, M))
    #initialize array for encoding quality values in the plot
    encoding_qualities = []
    #valid optimization methods
    methods = ["Analytical", "Numerical", "Random", "Benchmark"]
    #valid signal distributions
    signal_distributions = ["Uniform", "Random", "Half"]
    #initializing signal distribution
    p_s = np.full(M, 1/M)
    if signal_distr == "Uniform":
        p_s = np.full(M, 1/M)
    elif signal_distr == "Random":
        p_s = np.random.dirichlet(np.ones(M))
    elif signal_distr == "Half":
        p_temp_1 = np.full(M//2, 2/M)
        p_temp_2 = np.zeros(M//2)
        p_s = np.hstack((p_temp_1, p_temp_2))
    else:
        print("Error: Not a valid method from ")
        print(signal_distributions)
        return

    #neuron optimization steps in simulation
    for t in range(T):
        #randomized neuron optimization order per time step
        order = np.random.permutation(N)
        if method == "Analytical":
            #iterating through neuron k in the randomized order
            for k in order:
                strategies = optimize_neuron_analytical(N, M, k, strategies, mu, p_s, lr)
                strategy_traj[t, k, :] = strategies[k, :, 1]
        elif method == "Numerical":
            for k in order:
                strategies = optimize_neuron_numerical(N, M, k, strategies, mu, p_s, lr, neuron_types, kappa)
                strategy_traj[t, k, :] = strategies[k, :, 1]
        elif method == "Random":
            for k in order:
                strategies = optimize_neuron_random(N, M, k, strategies, mu, p_s, neuron_types)
                strategy_traj[t, k, :] = strategies[k, :, 1]
        elif method == "Benchmark":
            for k in order:
                strategies = optimize_neuron_benchmark(N, M, k, strategies, mu, p_s, lr)
                strategy_traj[t, k, :] = strategies[k, :, 1]
        else:
            print("Error: Not a valid method from ")
            print(methods)
            return
        # Measure encoding quality H(S | Y1..YN)
        joint_distr = joint_distribution(N, M, p_s, strategies)
        msg_distr = message_distribution(joint_distr)
        H_cond = conditional_entropy(N, M, joint_distr, msg_distr)
        encoding_qualities.append(H_cond)

    #print(f"Strategies ({method} for N = {N}, M = {M}, mu = {mu} \n {strategies}")

    #plots
    # plt.figure(figsize=(10, 5))
    # plt.plot(encoding_qualities, label='H(S | Y1..YN)')
    # plt.xlabel('Optimization Step')
    # plt.ylabel('Conditional Entropy')
    # plt.ylim(ymax=0)
    # plt.title(f'Encoding Quality Over Time ({method}) for N = {N}, M = {M}, mu = {mu}')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    return encoding_qualities, strategy_traj

def simulation_average(num_trials, N, M, T, mu, lr, signal_distr, method, neuron_types=[], kappa=-1.0):
    runs = []
    neuron_strategies = []
    for trial in range(num_trials):
        encoding_qualities, strategies = simulation(N, M, T, mu, lr, signal_distr, method, neuron_types, kappa)
        runs.append(encoding_qualities)
        neuron_strategies.append(strategies)
    runs = np.array(runs)
    neuron_strategies = np.array(neuron_strategies)
    mean_runs = runs.mean(axis=0)
    std_dev_runs = runs.std(axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(mean_runs, label='Average')
    for trial in range(num_trials-1):
        plt.plot(runs[trial], color="tab:blue", linestyle=":")
    plt.plot(runs[num_trials-1], color="tab:blue", linestyle=":", label="Trials")
    plt.xlabel('Optimization Step')
    plt.ylabel('Conditional Entropy')
    plt.ylim(ymax=0.1)
    plt.title(f'Encoding Quality Over Time ({method}) for {num_trials} Trials, N = {N}, M = {M}, mu = {mu}')
    if signal_distr == "Half":
        plt.title(f'Encoding Quality Over Time ({method}) for {num_trials} Trials, N = {N}, M = {M//2}, mu = {mu}')
    if kappa >= 0:
        plt.title(f'Encoding Quality Over Time ({method}) for {num_trials} Trials, N = {N}, M = {M}, mu = {mu}, kappa = {kappa}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(N, num_trials, figsize=(1.5*num_trials, 2*N), sharex=True, sharey=True)

    for k in range(N):
        for trial in range(num_trials):
            strategies = neuron_strategies[trial]
            if num_trials > 1:
                for s in range(M):
                    axes[k,trial].plot(strategies[:, k, s])#, linestyle=":")
                axes[k,trial].set_title(f"Neuron {k + 1}")
                axes[k,trial].set_ylim(-0.1, 1.1)
                axes[k,trial].set_xlabel("Step")
                if trial == 0:
                    axes[k, trial].set_ylabel("P(Y=1|s)")
            elif num_trials == 1:
                for s in range(M):
                    axes[k].plot(strategies[:, k, s])#, linestyle=":")
                axes[k].set_title(f"Neuron {k + 1} Strategy, Trial {trial+1}")
                axes[k].set_ylim(-0.1, 1.1)
                axes[k].set_xlabel("Optimization Step")
                if trial == 0:
                    axes[k].set_ylabel("P(Y=1|s)")
            else:
                print("Error")
                return
            #axes[k,s].legend()
    fig.suptitle(f'Neuron Strategies Over Time ({method}) for {num_trials} Trials, N = {N}, M = {M}, mu = {mu}')
    if signal_distr == "Half":
        fig.suptitle(f'Neuron Strategies Over Time ({method}) for {num_trials} Trials, N = {N}, M = {M//2}, mu = {mu}')
    if kappa >= 0:
        fig.suptitle(f'Neuron Strategies Over Time ({method}) for {num_trials} Trials, N = {N}, M = {M}, mu = {mu}, kappa = {kappa}')
    plt.tight_layout()
    plt.show()

    return

def simulation_closed_loop(N, M, T, mu, lr, signal_distr, method, neuron_types=[], kappa=-1.0):
    #each neuron k has a strategy matrix Y_k of shape (M, 2)
    #y_k[i, 1] = Pr(Y_k = 1 | s = s_i)
    #y_k[i, 0] = 1 - y_k[i, 1] = Pr(Y_k = 0 | s = s_i)
    #initialize random strategies
    strategies = np.random.rand(N, M)
    strategies = np.stack([1 - strategies, strategies], axis=2)  # shape (N, M, 2)
    strategy_traj = np.zeros((T, N, M))
    #initialize array for encoding quality values in the plot
    encoding_qualities = []
    #valid optimization methods
    methods = ["Analytical", "Numerical", "Random", "Benchmark"]
    #valid signal distributions
    signal_distributions = ["Uniform", "Random", "Half"]
    #initializing signal distribution
    p_s = np.full(M, 1 / M)
    if signal_distr == "Uniform":
        p_s = np.full(M, 1/M)
    elif signal_distr == "Random":
        p_s = np.random.dirichlet(np.ones(M))
    elif signal_distr == "Half":
        p_temp_1 = np.full(M//2, 2/M)
        p_temp_2 = np.zeros(M//2)
        p_s = np.hstack((p_temp_1, p_temp_2))
    else:
        print("Error: Not a valid method from ")
        print(signal_distributions)
        return

    #neuron optimization steps in simulation
    for t in range(T):


        # eigenvalues, eigenvectors = np.linalg.eig(Q_power)
        # for i in range(len(eigenvalues)):
        #     if math.isclose(1, eigenvalues[i].real, rel_tol = 0.01):
        #         eigenvalue = eigenvalues[i].real
        #         print(f"HERE: {eigenvalue}")
        # print(eigenvalues)
        # print(eigenvectors)

        # eigenvalue, eigenvector = eigs(Q_power, k=1, sigma=1)
        #
        # eigenvalue = eigenvalue.real
        # temp = np.zeros(len(eigenvector))
        # for i in range(len(eigenvector)):
        #     temp[i] = eigenvector[i][0].real
        # eigenvector = temp / np.sum(temp)
        # print("EIGEN")
        # print(eigenvalue)
        # print(temp)
        # print(eigenvector)
        # print(np.matmul(Q, eigenvector.T))

        #randomized neuron optimization order per time step
        order = np.random.permutation(N)
        if method == "Analytical":
            #iterating through neuron k in the randomized order
            for k in order:
                strategies = optimize_neuron_analytical(N, M, k, strategies, mu, p_s, lr)
                strategy_traj[t, k, :] = strategies[k, :, 1]
        elif method == "Numerical":
            for k in order:
                strategies = optimize_neuron_numerical(N, M, k, strategies, mu, p_s, lr, neuron_types, kappa)
                strategy_traj[t, k, :] = strategies[k, :, 1]
        elif method == "Random":
            for k in order:
                strategies = optimize_neuron_random(N, M, k, strategies, mu, p_s, neuron_types)
                strategy_traj[t, k, :] = strategies[k, :, 1]
        elif method == "Benchmark":
            for k in order:
                strategies = optimize_neuron_benchmark(N, M, k, strategies, mu, p_s, lr)
                strategy_traj[t, k, :] = strategies[k, :, 1]
        else:
            print("Error: Not a valid method from ")
            print(methods)
            return
        # Measure encoding quality H(S | Y1..YN)
        joint_distr = joint_distribution(N, M, p_s, strategies)
        msg_distr = message_distribution(joint_distr)
        H_cond = conditional_entropy(N, M, joint_distr, msg_distr)
        encoding_qualities.append(H_cond)

        # Compute Q and eigenvector
        Q = np.zeros((2 ** N, 2 ** N))
        for i in range(2 ** N):
            for j in range(2 ** N):
                # binary signal in array form
                msg_i = [(i >> index) & 1 for index in range(N)]
                msg_j = [(j >> index) & 1 for index in range(N)]
                entry = 1
                for neuron in range(N):
                    # print(f"Neuron {neuron+1} with output {msg_j[neuron]} for stimulus {(i, msg_i)} is value: {strategies[neuron, i, msg_j[neuron]]}")
                    entry *= strategies[neuron, i, msg_j[neuron]]
                # print(f"neuron output {msg_j} for stimulus input {msg_i} in array element {(i, j)} is value: {entry}")
                Q[i, j] = entry
        Q = normalize(Q, axis=1, norm="l1")
        print(f"Q: {Q}")
        # for i in range(2 ** N):
        #     print(sum(Q[i]))
        Q_power = Q
        for i in range(100):
            Q_power = np.matmul(Q_power, Q_power)
            Q_power = normalize(Q_power, axis=1, norm="l1")
        eigenvector = np.average(Q_power, axis=0)
        # eigenvector = eigenvector / np.sum(eigenvector)
        # print(f"Q power: {Q_power}")
        print(f"Eigenvector: {eigenvector}")
        #print(sum(eigenvector))

        p_s = eigenvector

    if kappa >= 0:
        print(f"Strategies ({method} for N = {N}, M = {M}, mu = {mu}, kappa = {kappa} \n {strategies}")
    else:
        print(f"Strategies ({method} for N = {N}, M = {M}, mu = {mu} \n {strategies}")

    #plots
    plt.figure(figsize=(10, 5))
    plt.plot(encoding_qualities, label='H(S | Y1..YN)')
    plt.xlabel('Optimization Step')
    plt.ylabel('Conditional Entropy')
    plt.ylim(ymax=0.1)
    plt.title(f'Encoding Quality Over Time ({method}) for N = {N}, M = {M}, mu = {mu}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(N, 1, figsize=(7, 2 * N), sharex=True, sharey=True)

    for k in range(N):
        for s in range(M):
            axes[k].plot(strategy_traj[:, k, s])  # , linestyle=":")
        axes[k].set_title(f"Neuron {k + 1} Strategy")
        axes[k].set_ylim(-0.1, 1.1)
        axes[k].set_xlabel("Optimization Step")
        axes[k].set_ylabel("P(Y=1|s)")
    fig.suptitle(f'Neuron Strategies Over Time ({method}), N = {N}, M = {M}, mu = {mu}')
    if signal_distr == "Half":
        fig.suptitle(
            f'Neuron Strategies Over Time ({method}), N = {N}, M = {M // 2}, mu = {mu}')
    if kappa >= 0:
        fig.suptitle(
            f'Neuron Strategies Over Time ({method}), N = {N}, M = {M}, mu = {mu}, kappa = {kappa}')
    plt.tight_layout()
    plt.show()

    return encoding_qualities, strategy_traj

#simulation_average(num_trials=10, N=4, M=4, T=500, mu=0.5, kappa=0.5, lr=0.01, signal_distr="Uniform", method="Numerical", neuron_types=["E", "E", "I", "I"])

#simulation_closed_loop(N=3, M=8, T=5, mu=2, lr=0.01, signal_distr="Uniform", method="Numerical")
#simulation_closed_loop(N=3, M=8, T=5, mu=2, lr=0.01, signal_distr="Uniform", method="Numerical")


simulation_closed_loop(N=3, M=8, T=100, mu=2, lr=0.01, signal_distr="Random", method="Numerical")
simulation_closed_loop(N=3, M=8, T=100, mu=2, lr=0.01, signal_distr="Uniform", method="Numerical")


# and then N=3, M=8, 10 runs, pick one very non-uniform stimulus 1/2, 1/4, 1/8


import skfuzzy as fuzz
import numpy as np
import random
import pandas as pd
import re
from qiskit_ibm_runtime.fake_provider import FakeMontrealV2 as FakeMontreal
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from evovaq.tools.operators import cx_uniform, sel_tournament, mut_gaussian
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.tools.utils import read_pkl_file, write_pkl_file

def graph_symmetry(G):
    list_degrees = G.degree
    degrees = []
    symmetry = True
    for node, degree in list_degrees:
        if (degree % 2) == 0:
            degrees.append('even')
        else:
            degrees.append('odd')
        if 'even' in degrees and 'odd' in degrees:
            symmetry = False
            break
    return symmetry, degrees

def test_ratio(c_max, c_max_qaoa, c_max_cluster):
    return (abs(c_max) - abs(c_max_qaoa)) / (abs(c_max) - abs(c_max_cluster))

def create_qaoa_circ(G, theta):
    nqubits = len(G.nodes())
    p = len(theta) // 2
    qc = QuantumCircuit(nqubits)

    beta = theta[:p]
    gamma = theta[p:]

    # initial_state
    for i in range(0, len(G.nodes)):
        qc.h(i)
    qc.barrier()

    for irep in range(0, p):

        # problem unitary
        for pair in list(G.edges()):
            qc.rzz(2 * gamma[irep], pair[0], pair[1])
        qc.barrier()

        # mixer unitary
        for i in range(0, len(G.nodes)):
            qc.rx(2 * beta[irep], i)
        if irep != 0:
            qc.barrier()
        else:
            pass

    qc.measure_all()

    return qc

def maxcut_obj(x, G):
    obj = 0
    x = x[::-1]
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1
    return obj
    
def compute_expectation(counts, G):
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = maxcut_obj(bitstring, G)
        avg += obj * count
        sum_count += count
    return avg / sum_count

def get_expectation_qiskit(G, backend, shots):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
    sampler = Sampler(backend)
    def execute_circuit(theta):
        qc = create_qaoa_circ(G, theta)
        qaoa_isa = pm.run(qc)
        pub = (qaoa_isa, theta)
        job = sampler.run([pub])
        pub_results = job.result()[0]
        counts = pub_results.data.meas.get_counts()
        return compute_expectation(counts, G)
    return execute_circuit
    
def clusteringFCM(features_train, features_test, fuzzifier):
    # Fuzzy C-Means
    best_run_for_n_clusters = {}
    for n_clusters in range(2, 11):
        jm_best = 100.0
        for n_run in range(1, 21):
            random_seed = 42 * (n_run - 1)

            cntr, u, u0, d, jm, p_runs, fpc = fuzz.cluster.cmeans(
                features_train.T, n_clusters, fuzzifier, error=1e-10, maxiter=1000, init=None, seed=random_seed)

            if jm.min() < jm_best:
                n_best_run = n_run
                jm_best = jm.min()
                cntr_best_run = cntr
                u_best_run = u
                u0_best_run = u0
                d_best_run = d
                jm_best_run = jm
                p_runs_best_run = p_runs
                fpc_best_run = fpc

        best_run_for_n_clusters[str(n_clusters)] = [n_best_run, cntr_best_run, u_best_run, u0_best_run, d_best_run,
                                                    jm_best_run, p_runs_best_run, fpc_best_run]
    best_fpc = 0.0
    best_n_clusters = 0
    dict_C_vs_fpc = {'C': [], 'fpc': []}
    for n_clust, results in list(best_run_for_n_clusters.items()):
        dict_C_vs_fpc['C'].append(n_clust)
        dict_C_vs_fpc['fpc'].append(results[-1])

        if results[-1] > best_fpc:
            best_fpc = results[-1]
            best_n_clusters = n_clust
        else:
            pass

    print(f"Best number of clusters {best_n_clusters} with fpc value {best_fpc}")

    n_best, cntr_best, u_best, u0_best, d_best, jm_best, p_runs_best, fpc_best = \
        best_run_for_n_clusters[str(best_n_clusters)]

    sorted_mu_clusters = np.argsort(u_best)
    sorted_train_instances_for_cluster = {str(i): [] for i in range(int(best_n_clusters))}

    for idx_cluster, sorted_index_cluster in zip(range(int(best_n_clusters)), sorted_mu_clusters):
        for idx_instance in sorted_index_cluster[::-1]:
            sorted_train_instances_for_cluster[str(idx_cluster)].append(
                [u_best[idx_cluster, idx_instance], idx_instance])

    u_test, u0_test, d_test, jm_test, p_runs_test, fpc_test = fuzz.cluster.cmeans_predict(
        features_test.T, cntr_best, fuzzifier, error=1e-10, maxiter=1000)

    max_mu_test_clusters = np.argmax(u_test, axis=0)

    queries_for_test_graph = []
    idx_train_instances_for_test_graph = []

    for idx_test_instance, idx_test_cluster in zip(range(len(features_test)), max_mu_test_clusters):

        sorted_train_instances = sorted_train_instances_for_cluster[str(idx_test_cluster)]
        idx_train_instances = []

        for mu, idx in sorted_train_instances:
            if mu >= u_test[idx_test_cluster, idx_test_instance]:
                idx_train_instances.append(idx)
            else:
                break

        if len(idx_train_instances) == 0:
            mu, idx = sorted_train_instances[0]
            idx_train_instances.append(idx)

        idx_train_instances_for_test_graph.append(idx_train_instances)
        queries_for_test_graph.append(len(idx_train_instances))

    return idx_train_instances_for_test_graph, queries_for_test_graph


def reusing_params_for_GA_training(X_train, X_test, idx_train_instances_for_test_graph, queries_for_test_graph, p,
                                   backend, pop_size, perc_pop_size, max_nfev):

    test_graphs = X_test['nx_obj'].tolist()
    c_maxs = X_test['c_max'].tolist()
    idx_graph = X_test['idx_graph']
    train_angles = X_train[f'p{p}_best_angles_QAOA'].to_numpy()
    r_1 = []
    r_3 = []

    for i, idx, G, idx_train_inst, tot_train_inst, c_max in zip(range(len(idx_graph)), idx_graph, test_graphs,
                                                                idx_train_instances_for_test_graph,
                                                                queries_for_test_graph, c_maxs):
        df = pd.read_excel( f'Results_QAOA/p={p}/n={len(G.nodes)}/{idx}.xlsx',
                            sheet_name='GA_MAX_EVAL=500', usecols=['n_run', 'best_cost'])
        selected_rows = df[df['n_run'] <= 10]
        c_ga = np.mean(selected_rows['best_cost'].to_numpy())

        symmetry, degrees = graph_symmetry(G)
        if symmetry:
            bounds_1 = [(-np.pi / 4, 0)]
            bounds_2 = [(-np.pi / 4, np.pi / 4) for _ in range(1, p)]
            bounds_3 = [(-np.pi / 2, np.pi / 2) for _ in range(p)]
        else:
            bounds_1 = [(-np.pi / 4, 0)]
            bounds_2 = [(-np.pi / 4, np.pi / 4) for _ in range(1, p)]
            bounds_3 = [(-np.pi, np.pi) for _ in range(p)]

        bounds = bounds_1 + bounds_2 + bounds_3
        cost_function = get_expectation_qiskit(G=G, backend=backend, shots=512)
        problem = Problem(2 * p, bounds, cost_function)

        optimizer = GA(selection=sel_tournament, crossover=cx_uniform, mutation=mut_gaussian, cxpb=0.7,
                       tournsize=3, cx_indpb=0.5, mut_indpb=0.25, mu=0, sigma=0.1)

        final_fcm_ga_costs = []
        best_fcm_ga = 100.0
        for n_run in range(10):
            if tot_train_inst < int(pop_size * perc_pop_size):
                np.random.seed(n_run)
                random.seed(n_run)
                rand_pop = problem.generate_random_pop(int(pop_size - tot_train_inst))
                arr_angles_FCM = [np.array(ang) for ang in train_angles[idx_train_inst]]
                FCM_pop = np.array(arr_angles_FCM)
                init_pop = np.concatenate((FCM_pop, rand_pop))
            else:
                arr_angles_FCM = [np.array(ang) for ang in
                                  train_angles[idx_train_inst[:int(pop_size * perc_pop_size)]]]
                FCM_pop = np.array(arr_angles_FCM)
                if len(FCM_pop) < pop_size:
                    np.random.seed(n_run)
                    random.seed(n_run)
                    rand_pop = problem.generate_random_pop(int(pop_size - pop_size * perc_pop_size))
                    init_pop = np.concatenate((FCM_pop, rand_pop))
                else:
                    init_pop = FCM_pop

            res = optimizer.optimize(problem, pop_size, initial_pop=init_pop, max_nfev=max_nfev, seed=n_run, n_run=n_run)
            final_fcm_ga_costs.append(res.fun)
            if res.fun < best_fcm_ga:
                best_fcm_ga = res.fun

        c_fcm_ga = np.mean(final_fcm_ga_costs)
        r_1.append(test_ratio(c_max, c_ga, c_fcm_ga))

        c_fcm = 100.0
        for angles in train_angles[idx_train_inst]:
            qc_res = create_qaoa_circ(G, angles)
            counts = backend.run(qc_res, seed_simulator=10, shots=512).result().get_counts()
            c_max_cluster = compute_expectation(counts, G)
            if c_max_cluster < c_fcm:
                c_fcm = c_max_cluster

        r_3.append(test_ratio(c_max, c_fcm, c_fcm_ga))

print(f"Test ratios for p={p}:\n r_1: {r_1} \n r_3: {r_3}")
print('Statistics r_1: ', np.mean(r_1), np.median(r_1), np.max(r_1), np.min(r_1))
print('Statistics r_3: ', np.mean(r_3), np.median(r_3), np.max(r_3), np.min(r_3))

return r_1, r_3


def main():
    pop_size = 10
    max_nfev = 510
    perc_pop_size = 1.0
    backend = FakeMontreal()
    X_train = read_pkl_file(f"datasets/db_train_GA.pkl")
    X_test = read_pkl_file(f"datasets/db_test_GA.pkl")
    features_train = X_train[['density', 'log_nodes', 'log_edges', 'log_first_largest_eigen',
                              'log_second_largest_eigen', 'log_ratio']].to_numpy()

    features_test = X_test[['density', 'log_nodes', 'log_edges', 'log_first_largest_eigen',
                            'log_second_largest_eigen', 'log_ratio']].to_numpy()

    idx_train_instances_for_test_graph, queries_for_test_graph = clusteringFCM(features_train, features_test, 2.0)
    for p in [3, 5, 7]:
        r_1, r_3 = reusing_params_for_GA_training(X_train, X_test, idx_train_instances_for_test_graph,
                                                         queries_for_test_graph, p, backend, pop_size, perc_pop_size,
                                                         max_nfev)

        r_1_stats = [np.mean(r_1), np.median(r_1), np.max(r_1), np.min(r_1)]
        impr_r_1 = np.sum(np.array(r_1) > 1.0) / len(r_1)
        r_1.extend(r_1_stats)
        r_1.append(impr_r_1)

        r_3_stats = [np.mean(r_3), np.median(r_3), np.max(r_3), np.min(r_3)]
        impr_r_3 = np.sum(np.array(r_3) > 1.0) / len(r_3)
        r_3.extend(r_3_stats)
        r_3.append(impr_r_3)

        perc = str(perc_pop_size).replace('.', '')
        write_pkl_file({'r_1': r_1, 'r_3': r_3}, f'p{p}_ratios_perc{perc}.pkl')


if __name__ == '__main__':
    main()

"""************************** local_top.py ****************"""
""" this script includs the local computations for single-shot ridge regression with decentralized statistic calculation """
""" Input args: --run json (this json structure may involve different field for different run) """ 
""" output: json """ 

## script name: local_top.py ##
## import dependent libraries ##
import argparse
import json
import numpy as np
import sys
import regression as reg

def local_1(args, computation_phase):
    input_list = args['input']
    X = input_list['covariates']
    y = input_list['dependents']
    lamb = input_list['lambda']
    biased_X = np.insert(X,0,1,axis=1)
    ## step 1 : generate the local beta_vector
    beta_vector = reg.one_shot_regression(X, y, lamb)

    ## step 2: generate the local fit statistics local r^2, t and p
    r_2 = reg.r_square(biased_X, y, beta_vector)
    ts_beta = reg.t_value(biased_X, y, beta_vector)
    dof = len(y) - len(beta_vector)
    ps_beta = reg.t_to_p(dof, ts_beta)

    ## step 3: generate the mean_y_local and count_local
    mean_y_local = np.mean(y)
    count_local = len(y)

    computation_output = json.dumps({'output':{'beta_vector_local': beta_vector.tolist(),
                                              'r_2_local' : r_2,
                                              'ts_local' : ts_beta.tolist(),
                                              'ps_local' : ps_beta,
                                              'mean_y_local' : mean_y_local,
                                              'count_local' : count_local,
                                              'computation_phase': computation_phase},
                                    'cache':{'covariates': X,
                                             'dependents': y,
                                             'lambda': lamb}
                                    },
                                    sort_keys=True, indent=4, separators=(',', ': '))
    return computation_output

def local_2(args, computation_phase):
    #After receiving  the mean_y_global, calculate the SSE_local, SST_local and varX_matrix_local
    cache_list = args['cache']
    input_list = args['input']
    X = cache_list['covariates']
    y = cache_list['dependents']
    avg_beta_vector = input_list['avg_beta_vector']
    biased_X = np.insert(X,0,1,axis=1)
    mean_y_global = input_list['mean_y_global']
    SSE_local = np.sum(np.square(y-y_estimate(biased_X, avg_beta_vector)))
    SST_local = np.sum(np.square(y-mean_y_global))
    varX_matrix_local = np.dot(biased_X.T, biased_X)

    computation_output = json.dumps({'output': {'SSE_local': SSE_local,
                                               'SST_local': SST_local,
                                               'varX_matrix_local': varX_matrix_local,
                                               'computation_phase': computation_phase}
                                   },
                                    sort_keys=True, indent=4, separators=(',', ': '))
    return computation_output


if __name__=='__main__':
   # read in coinstac args #
   parser = argparse.ArgumentParser(description='read in coinstac args for local computation')
   parser.add_argument('--run', type=str,  help='grab coinstac args')
   args = parser.parse_args()
   print(args.run)
   args.run = json.loads(args.run)
   input_list = args.run['input']

   #**********block 1***********#

   if 'computation_phase' not in input_list.keys():
       computation_phase = 'local_1' # computation_phase is the variable which controls the computation flow
       computation_output = local_1(args.run, computation_phase)
       sys.stdout.write(computation_output)

   #*********block 2***********#

   elif input_list['computation_phase'] == 'remote_1':
       computation_phase = 'local_2'
       computation_output = local_2(args.run, computation_phase)
       sys.stdout.write(computation_output)

   else:
       print("There are errors occured")

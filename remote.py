## script name : remote_top.py ##

## input dependent libraries ##
import numpy as np
import argparse
import json
import sys
import regression as reg


def remote_1(args, computation_phase):
    input_list = args['input']

    # step 1 calculate the averaged beta vector, mean_y_global and dof_global
    sum_beta_vector = np.zeros(np.shape(input_list[0]['beta_vector_local']))
    sum_y = 0
    sum_count = 0
    n_site = len(input_list)
    for i in range(0, n_site):
        sum_beta_vector = sum_beta_vector + input_list[i]['beta_vector_local']
        sum_y = sum_y + (input_list[i]['mean_y_local'])*(input_list[i]['count_local'])
        sum_count = sum_count + input_list[i]['count_local']
    avg_beta_vector = sum_beta_vector/n_site
    mean_y_global = sum_y/sum_count
    dof_global = sum_count - len(avg_beta_vector)

    ## step 2 retrieve the local fit statistics and save them in the cache
    beta_vector_local = []
    dof_local = []
    r_2_local = []
    ts_local = []
    ps_local = []

    for i in range(0, n_site):
        beta_vector_local.append(input_list[i]['beta_vector_local'])
        dof_local.append(input_list[i]['count_local']-len(avg_beta_vector))
        r_2_local.append(input_list[i]['r_2_local'])
        ts_local.append(input_list[i]['ts_local'])
        ps_local.append(input_list[i]['ps_local'])

    computation_output = json.dumps({'cache': {'avg_beta_vector': avg_beta_vector.tolist(),
                                               'mean_y_global': mean_y_global,
                                               'dof_global': dof_global,
                                               'dof_local': dof_local,
                                               'beta_vector_local': beta_vector_local,
                                               'r_2_local': r_2_local,
                                               'ts_local': ts_local,
                                               'ps_local': ps_local},
                                     'output': {'avg_beta_vector': avg_beta_vector.tolist(),
                                                'mean_y_global': mean_y_global,
                                                'computation_phase': computation_phase}
                                     }
                                    sort_keys=True, indent=4, separators=(',', ': '))
    return computation_output

 
def remote_2(args, computation_phase):
    ##  calculate the global model fit statistics,r_2_global, ts_global, ps_global
    cache_list = args['cache']
    input_list = args['input']
    avg_beta_vector = cache_list['avg_beta_vector']
    dof_global = cache_list['dof_global']
    n_site = len(input_list)
    SSE_global = 0
    SST_global = 0
    varX_matrix_global = []
    for i in range(0, n_site):
        SSE_global = SSE_global + input_list[i]['SSE_local']
        SST_global = SST_global + input_list[i]['SST_local']
        varX_matrix_global = varX_matrix_global + input_list[i]['varX_matrix_local']

    r_2_global = 1 - (SSE_global/SST_global)
    MSE = (1/dof_global)*SSE_global
    var_beta_global = MSE*(np.linalg.inv(varX_matrix_global).diagonal())
    se_beta_global = np.sqrt(var_beta_global)
    ts_global = avg_beta_vector / se_beta_global
    ps_global = reg.t_to_p(dof_global,ts_global)

    computation_output = json.dumps({'output':{'avg_beta_vector': cache_list['avg_beta_vector'],
                                               'beta_vector_local': cache_list['beta_vector_local'],
                                               'r_2_global': r_2_global,
                                               'ts_global': ts_global,
                                               'ps_global': ps_global,
                                               'r_2_local': cache_list['r_2_local'],
                                               'ts_local': cache_list['ts_local'],
                                               'ps_local': cache_list['ps_local'],
                                               'dof_global': cache_list['dof_global'],
                                               'dof_local': cache_list['dof_local'],  
                                               'complete': True}
                                    },
                                    sort_keys=True, indent=4, separators=(',', ': '))
    return computation_output


if __name__='__main__':
    # read in coinstac args
    parser = argparse.ArgumentParser(description='help read in coinstac input from local node')
    parser.add_argument('--run', type=str,  help='grab coinstac args')
    args = parser.parse_args()
    args.run = json.loads(args.run)

    #*******block 1*********#
    if input_list[0]['computation_phase'] == 'local_1':
        computation_phase = 'remote_1'
        remote_1(args.run, computation_phase)
        sys.stdout.write(computation_output)

    #*******block 2********#
    elif input_list[0]['computation_phase'] == 'local_2':
        computation_phase = 'remote_2'
        ## step 1 calculate the global model fit statistics,r_2_global, t_global, p_global
        remote_2(args.run, computation_phase)
        sys.stdout.write(computation_output)
    else:
        print("There are errors occured")

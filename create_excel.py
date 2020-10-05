import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--excel_path", type=str, default='./output.xlsx', help="excel file where the final results will be collated")
parser.add_argument("--chkpt_path", type=str, default='', help="checkpoint file name, this will be used to name the sheet")
opt = parser.parse_args()

exps = ['fib', 'nas_stea', 'nas_lob', 'nas_balloon']
exps_path='./experiments'
exp = 'fib'

columns =  ['A', 'B', 'C', 'D', 'E', 'F', 'G']
df = pd.DataFrame(columns=columns)
for exp in exps:
    f_name = exps_path + '/{:s}_baseline.txt'.format(exp)
    #temp_df = pd.read_csv(f_name, sep='\t', header=None).fillna(value = 0)
    if exp in 'nas_stea':
        print('nas_stea')
        temp_df = pd.read_csv(f_name, sep='\t', header=None, names=columns[:4]).fillna(value = 0)
    else:
        temp_df = pd.read_csv(f_name, sep='\t', header=None, names=columns).fillna(value = 0)

    if exp not in 'nas_stea':
        a = '{:s}'.format(exp)
        new_row =  pd.DataFrame({'A':[a], 'B':'acc', 'C':'avg_auc', 'D':'avg_ci', 'E':'auc0_ci0', 'F':'auc1_ci1', 'G':'auc2_ci2'})
    else:
        a = '{:s}'.format(exp)
        new_row = pd.DataFrame({'A':[a], 'B':'acc', 'C':'avg_auc', 'D':'avg_ci', 'E':'auc0_ci0'})

    temp_df = pd.concat([new_row, temp_df.iloc[:]]).reset_index(drop = True)
    # append an empty row
    #temp_df.append(pd.Series(), ignore_index=True)

    df = pd.concat([df, temp_df])
    df = df.append(pd.Series(), ignore_index=True)
    #df_prime = pd.concat([df, pd.DataFrame([[np.nan] * df.shape[1]], columns=df.columns)], ignore_index=True)
df.to_excel(opt.excel_path, index=False, sheet_name=opt.chkpt_path)
#writer.save()

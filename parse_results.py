import parse
import numpy as np


def parse_test_results(file_path, n_class=3):
    with open(file_path, 'r') as file:
        results = file.readline()
        Acc = parse.search('Acc: {:f}', results)[0]
        if n_class == 3:
            AUC_mean, CI_mean_lb, CI_mean_ub = parse.search('AUC: {:f} ({:f}, {:f})', results)
            AUC0, CI0_lb, CI0_ub = parse.search('AUC0: {:f} ({:f}, {:f})', results)
            AUC1, CI1_lb, CI1_ub = parse.search('AUC1: {:f} ({:f}, {:f})', results)
            AUC2, CI2_lb, CI2_ub = parse.search('AUC2: {:f} ({:f}, {:f})', results)
            return Acc, AUC_mean, [CI_mean_lb, CI_mean_ub], AUC0, [CI0_lb, CI0_ub], AUC1, [CI1_lb, CI1_ub], AUC2, [CI2_lb, CI2_ub]
        else:
            AUC, CI_lb, CI_ub = parse.search('AUC: {:f} ({:f}, {:f})', results)
            return Acc, AUC, [CI_lb, CI_ub]


def repeat_results():
    exp_names = ['fib', 'nas_stea', 'nas_lob', 'nas_balloon']
    n_classes = [3, 2, 3, 3]

    for exp_name, n_class in zip(exp_names, n_classes):
        for arch in ['10x_he', '10x_tri']:
            all_results = [
                {'acc': [], 'AUC_mean': [], 'CI_mean': [], 'AUC0': [], 'CI0': [], 'AUC1': [], 'CI1': [], 'AUC2': [],
                 'CI2': []},
                {'acc': [], 'AUC_mean': [], 'CI_mean': [], 'AUC0': [], 'CI0': [], 'AUC1': [], 'CI1': [], 'AUC2': [],
                 'CI2': []},
                {'acc': [], 'AUC_mean': [], 'CI_mean': [], 'AUC0': [], 'CI0': [], 'AUC1': [], 'CI1': [], 'AUC2': [],
                 'CI2': []}]
            for repeat in range(1, 8):
                arch_type = '{:s}{:d}'.format(arch, repeat)
                result_filepath = './experiments/results/{:s}_{:s}.txt'.format(exp_name, arch_type)
                with open(result_filepath, 'w') as file:
                    for i in range(3):
                        if n_class == 3:
                            Acc, AUC_mean, CI_mean, AUC0, CI0, AUC1, CI1, AUC2, CI2 = parse_test_results(
                                './experiments/{:s}/{:s}_{:d}/best/test_results.txt'.format(exp_name, arch_type, i + 1))
                            file.write(
                                '{:5.2f}\t{:5.2f} ({:5.2f}, {:5.2f})\t{:5.2f} ({:5.2f}, {:5.2f})\t{:5.2f} ({:5.2f}, {:5.2f})'
                                '\t{:5.2f} ({:5.2f}, {:5.2f})\n'.format(Acc, AUC_mean, CI_mean[0], CI_mean[1], AUC0,
                                                                        CI0[0], CI0[1],
                                                                        AUC1, CI1[0], CI1[1], AUC2, CI2[0], CI2[1]))
                            all_results[i]['acc'].append(Acc)
                            all_results[i]['AUC_mean'].append(AUC_mean)
                            all_results[i]['CI_mean'].append(CI_mean)
                            all_results[i]['AUC0'].append(AUC0)
                            all_results[i]['CI0'].append(CI0)
                            all_results[i]['AUC1'].append(AUC1)
                            all_results[i]['CI1'].append(CI1)
                            all_results[i]['AUC2'].append(AUC2)
                            all_results[i]['CI2'].append(CI2)
                        else:
                            Acc, AUC, CI = parse_test_results(
                                './experiments/{:s}/{:s}_{:d}/best/test_results.txt'.format(exp_name, arch_type, i + 1),
                                n_class=2)
                            file.write('{:5.2f}\t{:5.2f} ({:5.2f}, {:5.2f})\n'.format(Acc, AUC, CI[0], CI[1]))
                            all_results[i]['acc'].append(Acc)
                            all_results[i]['AUC_mean'].append(AUC)
                            all_results[i]['CI_mean'].append(CI)

            # compute average
            avg_result_filepath = './experiments/{:s}_{:s}_avg.txt'.format(exp_name, arch)
            with open(avg_result_filepath, 'w') as file:
                for i in range(3):
                    if n_class == 3:
                        Acc = np.mean(all_results[i]['acc'])
                        AUC_mean = np.mean(all_results[i]['AUC_mean'])
                        CI_mean = np.mean(all_results[i]['CI_mean'], axis=0)
                        AUC0 = np.mean(all_results[i]['AUC0'])
                        CI0 = np.mean(all_results[i]['CI0'], axis=0)
                        AUC1 = np.mean(all_results[i]['AUC1'])
                        CI1 = np.mean(all_results[i]['CI1'], axis=0)
                        AUC2 = np.mean(all_results[i]['AUC2'])
                        CI2 = np.mean(all_results[i]['CI2'], axis=0)
                        file.write(
                            '{:5.2f}\t{:5.2f} ({:5.2f}, {:5.2f})\t{:5.2f} ({:5.2f}, {:5.2f})\t{:5.2f} ({:5.2f}, {:5.2f})'
                            '\t{:5.2f} ({:5.2f}, {:5.2f})\n'.format(Acc, AUC_mean, CI_mean[0], CI_mean[1], AUC0, CI0[0],
                                                                    CI0[1], AUC1, CI1[0], CI1[1], AUC2, CI2[0], CI2[1]))
                    else:
                        Acc = np.mean(all_results[i]['acc'])
                        AUC = np.mean(all_results[i]['AUC_mean'])
                        CI = np.mean(all_results[i]['CI_mean'], axis=0)
                        file.write('{:5.2f}\t{:5.2f} ({:5.2f}, {:5.2f})\n'.format(Acc, AUC, CI[0], CI[1]))


def single_results():
    exp_names = ['fib', 'nas_stea', 'nas_lob', 'nas_balloon']
    n_classes = [3, 2, 3, 3]

    for exp_name, n_class in zip(exp_names, n_classes):
        #arch = 'baseline_he'
        arch = 'baseline'
        all_results = {'acc': [], 'AUC_mean': [], 'CI_mean': [], 'AUC0': [], 'CI0': [], 'AUC1': [], 'CI1': [], 'AUC2': [], 'CI2': []}
        result_filepath = './experiments/{:s}_{:s}.txt'.format(exp_name, arch)
        with open(result_filepath, 'w') as file:
            for i in range(3):
                if n_class == 3:
                    Acc, AUC_mean, CI_mean, AUC0, CI0, AUC1, CI1, AUC2, CI2 = parse_test_results(
                        './experiments/{:s}/{:s}_{:d}/best/test_results.txt'.format(exp_name, arch, i + 1))
                    file.write('Fold{:d}:\t{:5.2f}\t{:5.2f}\t({:5.2f}, {:5.2f})\t{:5.2f} ({:5.2f}, {:5.2f})'
                               '\t{:5.2f} ({:5.2f}, {:5.2f})\t{:5.2f} ({:5.2f}, {:5.2f})\n'
                               .format(i+1, Acc, AUC_mean, CI_mean[0], CI_mean[1], AUC0, CI0[0], CI0[1],
                                       AUC1, CI1[0], CI1[1], AUC2, CI2[0], CI2[1]))
                    all_results['acc'].append(Acc)
                    all_results['AUC_mean'].append(AUC_mean)
                    all_results['CI_mean'].append(CI_mean)
                    all_results['AUC0'].append(AUC0)
                    all_results['CI0'].append(CI0)
                    all_results['AUC1'].append(AUC1)
                    all_results['CI1'].append(CI1)
                    all_results['AUC2'].append(AUC2)
                    all_results['CI2'].append(CI2)
                else:
                    Acc, AUC, CI = parse_test_results('./experiments/{:s}/{:s}_{:d}/best/test_results.txt'
                                                      .format(exp_name, arch, i + 1), n_class=2)
                    file.write('Fold{:d}:\t{:5.2f}\t{:5.2f}\t({:5.2f}, {:5.2f})\n'.format(i+1, Acc, AUC, CI[0], CI[1]))
                    all_results['acc'].append(Acc)
                    all_results['AUC_mean'].append(AUC)
                    all_results['CI_mean'].append(CI)

            # compute average
            if n_class == 3:
                Acc = np.mean(all_results['acc'])
                AUC_mean = np.mean(all_results['AUC_mean'])
                CI_mean = np.mean(all_results['CI_mean'], axis=0)
                AUC0 = np.mean(all_results['AUC0'])
                CI0 = np.mean(all_results['CI0'], axis=0)
                AUC1 = np.mean(all_results['AUC1'])
                CI1 = np.mean(all_results['CI1'], axis=0)
                AUC2 = np.mean(all_results['AUC2'])
                CI2 = np.mean(all_results['CI2'], axis=0)
                file.write('Avg:\t{:5.2f}\t{:5.2f}\t({:5.2f}, {:5.2f})\t{:5.2f} ({:5.2f}, {:5.2f})\t{:5.2f} ({:5.2f}, {:5.2f})'
                           '\t{:5.2f} ({:5.2f}, {:5.2f})\n'.format(Acc, AUC_mean, CI_mean[0], CI_mean[1], AUC0, CI0[0],
                                                                   CI0[1], AUC1, CI1[0], CI1[1], AUC2, CI2[0], CI2[1]))
            else:
                Acc = np.mean(all_results['acc'])
                AUC = np.mean(all_results['AUC_mean'])
                CI = np.mean(all_results['CI_mean'], axis=0)
                file.write('Avg:\t{:5.2f}\t{:5.2f}\t({:5.2f}, {:5.2f})\n'.format(Acc, AUC, CI[0], CI[1]))

single_results()
#repeat_results()

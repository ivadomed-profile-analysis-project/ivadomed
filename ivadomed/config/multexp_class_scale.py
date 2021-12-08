import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
import time


def multbarplot(X, data, label, width, err, ylabel, title, curr_subplot):
    plt.subplot(2, 2, curr_subplot)

    X_axis = np.arange(len(X))
    n = len(data)
    # plt.figure(figsize=(10, 10))

    for i in range(n):
        curr_data = data[i]
        curr_label = label[i]
        curr_err = err[i]
        plt.bar(X_axis - width / 2 + (i * width / n), curr_data, width / n, label=curr_label, align='edge')
               # yerr=curr_err)

    plt.xticks(X_axis, X)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

# Used this guide to add bar counts to stacked bar plots.
# https://www.pythoncharts.com/matplotlib/stacked-bar-charts-labels/
def stackedbarplot(X, data, label, width, ylabel, title, filename):
    X_axis = np.arange(len(X))
    n = len(data)
    fig, ax = plt.subplots(figsize=(20, 15))


    bot = [0 for i in range(len(X))]
    for i in range(n):
        curr_data = data[i]
        curr_label = label[i]
        plt.bar(X_axis, curr_data, width / n, label=curr_label, bottom=bot)
        bot = [x + y for x, y in zip(bot, curr_data)]

    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            round(bar.get_height(), 3),
            ha='center',
            weight='bold'
        )

    #ax.set_yscale('log')

    plt.xticks(X_axis, X)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    # plt.show()


def trainsubplot(mod1trn, mod2trn, mod3trn, label, ylabel, title, curr_subplot):
    plt.subplot(2, 3, curr_subplot)
    len_1 = mod1trn.shape[0]
    len_2 = mod2trn.shape[0]
    len_3 = mod3trn.shape[0]

    plt.plot(range(len_1), mod1trn, label=label[0])
    plt.plot(range(len_2), mod2trn, label=label[1])
    plt.plot(range(len_3), mod3trn, label=label[2])

    plt.xlabel('Mini-batch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()


def main():
    # Get path parameters for ivadomed and experiment parameters.
    numexp = 3
    # Replace this with your own directory.
    # path = '/home/sshatagopam/ivadomed/'
    path = 'C:/Users/harsh/ivadomed/'

    direxp = path + 'experiments/class_scale/'
    dirpath = [
        direxp + 'resnet_0.5/',
        direxp + 'resnet_1.0/',
        direxp + 'resnet_10.0/'
    ]

    dirsave = path + 'experiments/final/class_scale/'
    savepath = [
        dirsave + 'resnet_0.5/',
        dirsave + 'resnet_1.0/',
        dirsave + 'resnet_10.0/'
    ]

    plotdir = path + 'experiments/plots/class_scale/'

    modelnames = [
        'resnet_0.5',
        'resnet_1.0',
        'resnet_10.0'
    ]
    config_path_0_5x = path + 'ivadomed/config/class_0.5xdata/'
    config_path_1x = path + 'ivadomed/config/class_1xdata/'
    config_path_2x = path + 'ivadomed/config/class_10xdata/'
    config_files = [
        config_path_0_5x + 'resnet_0.5xdata.json',
        config_path_1x + 'resnet_1xdata.json',
        config_path_2x + 'resnet_10xdata.json'
    ]
    # Create directories if they don't exist yet.
    for dir in dirpath:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for dir in savepath:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

    if not os.path.exists(config_path_1x):
        os.makedirs(config_path_1x)

    if not os.path.exists(config_path_2x):
        os.makedirs(config_path_2x)

    if not os.path.exists(config_path_0_5x):
        os.makedirs(config_path_0_5x)

    # Make lists for storing temp profiling data.
    time_data = []
    time_train = []
    time_post = []

    gpu_util_mean = []
    gpu_util_err = []

    cpu_util_mean = []
    cpu_util_err = []

    gpu_mem_mean = []
    gpu_mem_err = []

    cpu_mem_mean = []
    cpu_mem_err = []

    # We repeat experiments for each architecture.
    for path_num in range(len(dirpath)):
        # Get current experiment parameters.
        currdir = dirpath[path_num]
        final_path = savepath[path_num]
        curr_model = modelnames[path_num]
        curr_config = config_files[path_num]

        tlog = []
        vlog = []
        slog = []

        # Append path for each individual experiment csv file.
        for i in range(numexp):
            tlog.append(currdir + 'trainlog' + str(i + 1) + '.csv')
            vlog.append(currdir + 'vallog' + str(i + 1) + '.csv')
            slog.append(currdir + 'syslog' + str(i + 1) + '.csv')

        # Get the commands for each ivadomed call (per experiment).
        command = []
        for i in range(numexp):
            com = ['ivadomed', '-c', curr_config, '--tlog', tlog[i], '--vlog', vlog[i], '--slog', slog[i]]
            command.append(com)

        # Run the current experiment numexp times.
        for i in range(numexp):
            subprocess.run(command[i])

        sysdf = []
        trndf = []
        valdf = []

        # Create csv files for system, train, and validation files.
        for i in range(numexp):
            sysdf.append(pd.DataFrame(pd.read_csv(slog[i])))
            trndf.append(pd.DataFrame(pd.read_csv(tlog[i])))
            valdf.append(pd.DataFrame(pd.read_csv(vlog[i])))

        sys = pd.DataFrame(columns=['comp', 'time', 'gpu_util', 'cpu_util', 'gpu_mem', 'main_mem'])
        trn = pd.DataFrame(columns=['time', 'train_gpu_util', 'train_cpu_util', 'train_gpu_mem', 'train_main_mem', 'train_loss'])
        val = pd.DataFrame(columns=['time', 'val_gpu_util', 'val_cpu_util', 'val_gpu_mem', 'val_main_mem', 'val_loss'])

        time_row = [df.iloc[0, 1] for df in sysdf]
        time_data.append(np.mean(time_row))
        time_row = [df.iloc[1, 1] for df in sysdf]
        time_train.append(np.mean(time_row))
        time_row = [df.iloc[2, 1] for df in sysdf]
        time_post.append(np.mean(time_row))

        curr_exp_gpu_util_mean = []
        curr_exp_gpu_util_err = []

        curr_exp_cpu_util_mean = []
        curr_exp_cpu_util_err = []

        curr_exp_gpu_mem_mean = []
        curr_exp_gpu_mem_err = []

        curr_exp_cpu_mem_mean = []
        curr_exp_cpu_mem_err = []

        for i in range(sysdf[0].shape[0]):

            gpu_util_row = [df.iloc[i, 2] for df in sysdf]
            curr_exp_gpu_util_mean.append(np.mean(gpu_util_row))
            curr_exp_gpu_util_err.append(np.std(gpu_util_row))

            cpu_util_row = [df.iloc[i, 3] for df in sysdf]
            curr_exp_cpu_util_mean.append(np.mean(cpu_util_row))
            curr_exp_cpu_util_err.append(np.std(cpu_util_row))

            gpu_mem_row = [df.iloc[i, 4] for df in sysdf]
            curr_exp_gpu_mem_mean.append(np.mean(gpu_mem_row))
            curr_exp_gpu_mem_err.append(np.std(gpu_mem_row))

            cpu_mem_row = [df.iloc[i, 5] for df in sysdf]
            curr_exp_cpu_mem_mean.append(np.mean(cpu_mem_row))
            curr_exp_cpu_mem_err.append(np.std(cpu_mem_row))

            time_sum = gpu_util_sum = cpu_util_sum = gpu_mem_sum = main_mem_sum = 0

            for j in range(numexp):
                time_sum += sysdf[j].iloc[i, 1]
                gpu_util_sum += sysdf[j].iloc[i, 2]
                cpu_util_sum += sysdf[j].iloc[i, 3]
                gpu_mem_sum += sysdf[j].iloc[i, 4]
                main_mem_sum += sysdf[j].iloc[i, 5]
                comp = sysdf[j].iloc[i, 0]

            time_sum /= numexp
            gpu_util_sum /= numexp
            cpu_util_sum /= numexp
            gpu_mem_sum /= numexp
            main_mem_sum /= numexp

            sys = sys.append({'comp': comp, 'time': time_sum, 'gpu_util': gpu_util_sum, 'cpu_util': cpu_util_sum,
                              'gpu_mem': gpu_mem_sum, 'main_mem': main_mem_sum}, ignore_index=True)

        # time_mean.append(curr_exp_time_mean)

        gpu_util_mean.append(curr_exp_gpu_util_mean)
        gpu_util_err.append(curr_exp_gpu_util_err)

        cpu_util_mean.append(curr_exp_cpu_util_mean)
        cpu_util_err.append(curr_exp_cpu_util_err)

        gpu_mem_mean.append(curr_exp_gpu_mem_mean)
        gpu_mem_err.append(curr_exp_gpu_mem_err)

        cpu_mem_mean.append(curr_exp_cpu_mem_mean)
        cpu_mem_err.append(curr_exp_cpu_mem_err)

        for i in range(trndf[0].shape[0]):
            time_sum = gpu_util_sum = cpu_util_sum = gpu_mem_sum = main_mem_sum = loss_sum = 0

            for j in range(numexp):
                time_sum += trndf[j].iloc[i, 0]
                gpu_util_sum += trndf[j].iloc[i, 1]
                cpu_util_sum += trndf[j].iloc[i, 2]
                gpu_mem_sum += trndf[j].iloc[i, 3]
                main_mem_sum += trndf[j].iloc[i, 4]
                loss_sum += trndf[j].iloc[i, 5]

            time_sum /= numexp
            gpu_util_sum /= numexp
            cpu_util_sum /= numexp
            gpu_mem_sum /= numexp
            main_mem_sum /= numexp
            loss_sum /= numexp

            trn = trn.append({'time': time_sum, 'train_gpu_util': gpu_util_sum, 'train_cpu_util': cpu_util_sum,
                              'train_gpu_mem': gpu_mem_sum, 'train_main_mem': main_mem_sum, 'train_loss': loss_sum},
                             ignore_index=True)

        for i in range(valdf[0].shape[0]):
            time_sum = gpu_util_sum = cpu_util_sum = gpu_mem_sum = main_mem_sum = loss_sum = 0

            for j in range(numexp):
                time_sum += valdf[j].iloc[i, 0]
                gpu_util_sum += valdf[j].iloc[i, 1]
                cpu_util_sum += valdf[j].iloc[i, 2]
                gpu_mem_sum += valdf[j].iloc[i, 3]
                main_mem_sum += valdf[j].iloc[i, 4]
                loss_sum += valdf[j].iloc[i, 5]

            time_sum /= numexp
            gpu_util_sum /= numexp
            cpu_util_sum /= numexp
            gpu_mem_sum /= numexp
            main_mem_sum /= numexp
            loss_sum /= numexp

            val = val.append({'time': time_sum, 'val_gpu_util': gpu_util_sum, 'val_cpu_util': cpu_util_sum,
                              'val_gpu_mem': gpu_mem_sum, 'val_main_mem': main_mem_sum, 'val_loss': loss_sum},
                             ignore_index=True)

        sys.to_csv(final_path + curr_model + '_sys.csv', encoding='utf-8', index=False)
        trn.to_csv(final_path + curr_model + '_trn.csv', encoding='utf-8', index=False)
        val.to_csv(final_path + curr_model + '_val.csv', encoding='utf-8', index=False)

    # SYSTEM SUBPLOT

    X = ['Data', 'Train', 'Post']
    label = ['ResNet, dsize=0.5',
             'ResNet, dsize=1.0',
             'ResNet, dsize=10.0'
             ]
    width = 0.75

    fig, ax = plt.subplots(figsize=(20, 15))
    fig.tight_layout(pad=5)

    ylabel = 'GPU Utilization (%)'
    title = 'GPU Utilization Across Dataset Sizes'
    multbarplot(X=X, data=gpu_util_mean, label=label, width=width, err=gpu_util_err, ylabel=ylabel,
                title=title, curr_subplot=1)

    ylabel = 'CPU Utilization (%)'
    title = 'CPU Utilization Across Dataset Sizes'
    multbarplot(X=X, data=cpu_util_mean, label=label, width=width, err=cpu_util_err, ylabel=ylabel,
                title=title, curr_subplot=2)

    ylabel = 'GPU Memory Utilization (%)'
    title = 'GPU Memory Utilization Across Dataset Sizes'
    multbarplot(X=X, data=gpu_mem_mean, label=label, width=width, err=gpu_mem_err, ylabel=ylabel,
                title=title, curr_subplot=3)

    ylabel = 'CPU Memory Utilization (%)'
    title = 'CPU Memory Utilization Across Dataset Sizes'
    multbarplot(X=X, data=cpu_mem_mean, label=label, width=width, err=cpu_mem_err, ylabel=ylabel,
                title=title, curr_subplot=4)

    plt.savefig(plotdir + 'subplot.png')
    # plt.show()

    # SYSTEM TIME PLOT

    X = ['ResNet, dsize=0.5',
         'ResNet, dsize=1.0',
         'ResNet, dsize=10.0'
        ]
    time_mean = [time_data, time_train, time_post]
    label = ['Data', 'Train', 'Post']
    filename = plotdir + 'time_per_comp_plot.png'
    ylabel = 'Time (seconds)'
    title = 'Time per Pipeline Component Across Dataset Sizes'
    stackedbarplot(X=X, data=time_mean, label=label, width=2, ylabel=ylabel, title=title, filename=filename)

    # TRAINING SUBPLOT
    final_path = path + 'experiments/final/class_scale/'
    _0_5_trn = pd.read_csv(final_path + 'resnet_0.5/' + 'resnet_0.5_trn.csv')
    _1_trn = pd.read_csv(final_path + 'resnet_1.0/' + 'resnet_1.0_trn.csv')
    _2_trn = pd.read_csv(final_path + 'resnet_10.0/' + 'resnet_10.0_trn.csv')

    fig, ax = plt.subplots(2,3, figsize=(20, 15))
    label = ['ResNet, dsize=0.5',
             'ResNet, dsize=1.0',
             'ResNet, dsize=10.0'
        ]

    ylabel = 'GPU Utilization (%)'
    title = 'Training GPU Utilization Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_trn.iloc[:, 1], _1_trn.iloc[:, 1], _2_trn.iloc[:, 1], label, ylabel, title, 1)

    ylabel = 'CPU Utilization (%)'
    title = 'Training CPU Utilization Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_trn.iloc[:, 2], _1_trn.iloc[:, 2], _2_trn.iloc[:, 2], label, ylabel, title, 2)

    ylabel = 'GPU Memory Utilization (%)'
    title = 'Training GPU Memory Utilization Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_trn.iloc[:, 3], _1_trn.iloc[:, 3], _2_trn.iloc[:, 3], label, ylabel, title, 3)

    ylabel = 'CPU Memory Utilization (%)'
    title = 'Training CPU Memory Utilization Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_trn.iloc[:, 4], _1_trn.iloc[:, 4], _2_trn.iloc[:, 4], label, ylabel, title, 4)

    ylabel = 'Time (s)'
    title = 'Training Time Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_trn.iloc[:, 0], _1_trn.iloc[:, 0], _2_trn.iloc[:, 0], label, ylabel, title, 5)

    ylabel = 'Log Loss'
    title = 'Log Loss Per Mini-Batch Across Architectures'
    trainsubplot(_0_5_trn.iloc[:, 5], _1_trn.iloc[:, 5], _2_trn.iloc[:, 5], label, ylabel, title, 6)

    plt.savefig(plotdir + 'training_subplot.png')

    # VALIDATION SUBPLOT

    _0_5_val = pd.read_csv(final_path + 'resnet_0.5/' + 'resnet_0.5_val.csv')
    _1_val = pd.read_csv(final_path + 'resnet_1.0/' + 'resnet_1.0_val.csv')
    _2_val = pd.read_csv(final_path + 'resnet_10.0/' + 'resnet_10.0_val.csv')

    fig, ax = plt.subplots(2,3, figsize=(20, 15))
    label = ['ResNet, dsize=0.5',
             'ResNet, dsize=1.0',
             'ResNet, dsize=10.0',
             ]

    ylabel = 'GPU Utilization (%)'
    title = 'Validation GPU Utilization Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_val.iloc[:, 1], _1_val.iloc[:, 1], _2_val.iloc[:, 1], label, ylabel, title, 1)

    ylabel = 'CPU Utilization (%)'
    title = 'Validation CPU Utilization Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_val.iloc[:, 2], _1_val.iloc[:, 2], _2_val.iloc[:, 2], label, ylabel, title, 2)

    ylabel = 'GPU Memory Utilization (%)'
    title = 'Validation GPU Memory Utilization Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_val.iloc[:, 3], _1_val.iloc[:, 3], _2_val.iloc[:, 3], label, ylabel, title, 3)

    ylabel = 'CPU Memory Utilization (%)'
    title = 'Validation CPU Memory Utilization Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_val.iloc[:, 4], _1_val.iloc[:, 4], _2_val.iloc[:, 4], label, ylabel, title, 4)

    ylabel = 'Time (s)'
    title = 'Validation Time Per Mini-Batch Across Dataset Sizes'
    trainsubplot(_0_5_val.iloc[:, 0], _1_val.iloc[:, 0], _2_val.iloc[:, 0], label, ylabel, title, 5)

    ylabel = 'Log Loss'
    title = 'Log Loss Per Mini-Batch Across Architectures'
    trainsubplot(_0_5_val.iloc[:, 5], _1_val.iloc[:, 5], _2_val.iloc[:, 5], label, ylabel, title, 6)


    plt.savefig(plotdir + 'validation_subplot.png')



if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\n\nScript executed in {end - start} seconds.")

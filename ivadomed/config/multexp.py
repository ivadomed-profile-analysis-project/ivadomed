import subprocess
import pandas as pd

numexp = 3
dirpath = 'C:/Users/harsh/ivadomed/experiments/'

tlog = []
vlog = []
slog = []

for i in range(numexp):
    tlog.append(dirpath + 'trainlog' + str(i+1) + '.csv')
    vlog.append(dirpath + 'vallog' + str(i+1) + '.csv')
    slog.append(dirpath + 'syslog' + str(i+1) + '.csv')

command = []
for i in range(numexp):
    com = ['ivadomed', '-c', 'config.json', '--tlog', tlog[i], '--vlog', vlog[i], '--slog', slog[i]]
    command.append(com)

for i in range(numexp):
    subprocess.run(command[i])



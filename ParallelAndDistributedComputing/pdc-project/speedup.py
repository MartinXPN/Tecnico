import subprocess
import os
from sys import argv
from distutils.util import strtobool

CHECK_RESULTS = bool(strtobool(argv[1].lower()))
NUM_THREADS = int(argv[2])
MAX_NUM_THREADS = int(argv[3])
NUM_THREADS_STEP = int(argv[4])

SEP = '--------------------------------------------------------'


def execution_time(command, input_path):
  process = subprocess.Popen([command, input_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  std = process.communicate()
  stdout = std[0].decode('utf-8')
  stderror = std[1].decode('utf-8')
  
  if CHECK_RESULTS:
    file = open(input_path[:-2]+'out', 'r')
    refoutput = file.read()
    file.close()
    if stdout != refoutput: 
      print('\033[0;31m{}'.format(stdout))
      print(refoutput)
      raise ValueError('\033[1;31mThe output of {} with instance {} is not correct!\033[0m'.format(command, input_path))

  time_index = stderror.find("Execution time: ") + 16
  return stderror[time_index: time_index+5]


subprocess.call('make', stdout=subprocess.PIPE)

folder = 'instances'

if argv[5] == 'all':
  instances = ['inst0', 'inst1', 'inst2', 'inst30-40-10-2-10', 'inst500-500-20-2-100', 'inst1000-1000-100-2-30']
elif argv[5] == 'allBig':
  folder = 'instances-larger'
  instances = ['inst200-10000-50-100-300','inst400-50000-30-200-500','inst600-10000-10-40-400','instML100k','instML1M','inst50000-5000-100-2-5']
else:
  instances = argv[5:]

print('\033[1;34mNUM_THREADS\tSerial\tParallel Speedup\n{}\033[0m'.format(SEP))

for instance in instances:
  print('\033[1;36m{}\033[0m'.format(instance))

  path = '{}/{}.in'.format(folder, instance)
  serial_time = float(execution_time('./matFact', path))

  for i in range(NUM_THREADS, MAX_NUM_THREADS + 1, NUM_THREADS_STEP):
    os.environ["OMP_NUM_THREADS"] = str(i)
    parallel_time = float(execution_time('./matFact-omp', path))
    speedup = float(serial_time) / float(parallel_time)
    print('{}{}{}{}'.format(str(i).ljust(16), str(serial_time).ljust(8), str(parallel_time).ljust(9), str(speedup).strip().ljust(15)))
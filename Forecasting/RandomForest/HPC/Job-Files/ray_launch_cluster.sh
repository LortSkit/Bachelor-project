#!/bin/sh
#BSUB -J hyptun
#BSUB -q hpc
#BSUB -W 168:00
#BSUB -n 100
#BSUB -R "span[block=2]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -o hyptun_%J.out
#BSUB -e hyptun_%J.err
#BSUB -Ne

cd ~s204092
source bach-env/bin/activate
cd ~s204092/Bachelor-project/Forecasting/RandomForest

#Very much stolen from takaomoriyama on github (lots of love <3) https://github.com/IBMSpectrumComputing/ray-integration/blob/main/ray_launch_cluster.sh

#bias to selection of higher range ports
function getfreeport()
{
    CHECK="do while"
    while [[ ! -z $CHECK ]]; do
        port=$(( ( RANDOM % 40000 )  + 20000 ))
        CHECK=$(netstat -a | grep $port)
    done
    echo $port
}

while getopts ":c:n:m:" option;do
    case "${option}" in
    c) c=${OPTARG}
        user_command=$c
    ;;
    n) n=${OPTARG}
        conda_env=$n
    ;;
    m) m=${OPTARG}
        object_store_mem=$m
    ;;
    *) echo "Did not supply the correct arguments"
    ;;
    esac
    done


hosts=()
for host in `cat $LSB_DJOB_HOSTFILE | uniq`
do
        echo "Adding host: " $host
        hosts+=($host)
done

echo "The host list is: " "${hosts[@]}"

port=$(getfreeport)
echo "Head node will use port: " $port

export port

dashboard_port=$(getfreeport)
echo "Dashboard will use port: " $dashboard_port


echo "Num cpus per host is:" $LSB_MCPU_HOSTS
IFS=' ' read -r -a array <<< "$LSB_MCPU_HOSTS"
declare -A associative
i=0
len=${#array[@]}
while [ $i -lt $len ]
do
    key=${array[$i]}
    value=${array[$i+1]}
    associative[$key]+=$value
    i=$((i=i+2))
done

num_gpu=0
if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
    num_gpu=0
else
    num_gpu=`echo $CUDA_VISIBLE_DEVICES | tr , ' ' | wc -w`
fi

#Assumption only one head node and more than one 
#workers will connect to head node

head_node=${hosts[0]}

export head_node

echo "Object store memory for the cluster is set to 8GB"

echo "Starting ray head node on: " ${hosts[0]}

if [ -z $object_store_mem ]
then
    echo "using default object store mem of 8GB make sure your cluster has mem greater than 8GB"
    object_store_mem=8000000000
else
    echo "The object store memory in bytes is: " "$object_store_mem"
fi

num_cpu_for_head=${associative[$head_node]}

command_launch="blaunch -z ${hosts[0]} ray start --head --node-ip-address=$head_node --port $port --dashboard-port $dashboard_port --num-cpus $num_cpu_for_head --num-gpus $num_gpu --object-store-memory $object_store_mem"

$command_launch &



sleep 20

command_check_up="ray status --address $head_node:$port"

while ! $command_check_up
do
    sleep 3
done



workers=("${hosts[@]:1}")

echo "adding the workers to head node: " "${workers[*]}"
#run ray on worker nodes and connect to head
for host in "${workers[@]}"
do
    echo "starting worker on: " $host "and using master node: " $head_node

    sleep 10
    num_cpu=${associative[$host]}
    command_for_worker="blaunch -z $host ray  start --address $head_node:$port --num-cpus $num_cpu --num-gpus $num_gpu --object-store-memory $object_store_mem"
    
    
    $command_for_worker &
    sleep 10
    command_check_up_worker="blaunch -z $host ray  status --address $head_node:$port"
    while ! $command_check_up_worker
    do
        sleep 3
    done
done

#Run workload
#eg of user workload python sample_code_for_ray.py
echo "Running user workload: "
python nbrun.py


if [ $? != 0 ]; then
    echo "Failure: " $?
    exit $?
else
    echo "Done."
fi
echo "Shutting down the Job"
bkill $LSB_JOBID
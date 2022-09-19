#!/bin/bash

echo $@
conda_environment_export=$1
python_script=$2

empty="_empty_"

Y=($@)
python_arguments=${Y[@]:2}
# echo "${!python_arguments[@]}"
# #echo "${python_arguments[@]}"
# COUNTER_ARGUMENTS=1
# for i in $@
# do
#     echo  ${python_arguments[$COUNTER_ARGUMENTS]}
#     echo  ${python_arguments[$COUNTER_ARGUMENTS]}
#     empty='_empty_'
#         if [[ "${python_arguments[$COUNTER_ARGUMENTS]}" == "$empty" ]]; then
#                 python_arguments[$COUNTER_ARGUMENTS]=""
#         fi
#     let COUNTER_ARGUMENTS+=1
# #    echo  python_arguments[$i]
# done

# empty="_empty_"
# python_arguments=()
# Y=($@)
# COUNTER_ARGUMENTS=1
# for python_argument in $@; do
#    if [ $COUNTER_ARGUMENTS -gt 2 ]
#    then
#      # if [ `expr $COUNTER_ARGUMENTS % 2` == 0 ]
#      # then
#      #  python_arguments=`echo $python_arguments '"'$i'"'`
#      # else
#      echo $python_argument
#      echo $empty
#       if [[ "$python_argument" == "$empty" ]]
#       then
#         python_argument=""
#       fi
#     echo $python_argument
#     python_arguments+=($python_argument)
#
#      # fi
#    fi
#    let COUNTER_ARGUMENTS+=1
#
# done

# echo conda init bash
# conda init bash
# echo conda activate KLIMPALA $conda_environment
# conda activate KLIMPALA $conda_environment
# /home/woutersh/software/anaconda3/envs/KLIMPALA/etc/profile.d/conda.sh
export PATH="$1:$PATH"
echo which python
which python
echo python_script : $python_script
echo python_arguments : "$python_arguments"
echo python $python_script ${python_arguments[@]}
python $python_script ${python_arguments[@]}


#!/bin/bash

#usage, arguments =  starting batch size, interval count, end count
#


counter=$1

while [ $counter -lt $3 ]
do
	echo "MLP_constant $counter"
	echo "------------"
    python3 MLP_test.py $counter 
    # echo "MLP_invscaling $counter"
    # echo "------------"
    # python3 MLP_test.py $counter 
    # echo "MLP_invscaling $counter"
    # echo "------------"
    # python3 MLP_test.py $counter 

    counter=$(($counter+$2))
    # echo "BNB $counter"
done


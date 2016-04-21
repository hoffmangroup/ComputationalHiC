#!/bin/bash
#
#$ -cwd
#$ -S /bin/bash
#
set -o nounset -o pipefail -o errexit
set -o xtrace

label="optim_HeldOutSepa_saveMse_2016-04-21"
chromSel="chr21"

randomNumber=$(shuf -i1-10000 -n1)
folderName="../results/tests_${chromSel}_${label}"
mkdir $folderName
outputFileName="$folderName/multiple_output_${chromSel}_rand${randomNumber}"
echo $outputFileName


dataSource="Thurman_Miriam"
trainSamplesPerc=50
trainNegElemsPerc=50
trainTupleLimit=3000

valNegElemsPerc=90
valTupleLimit=3000

trainExecutionMode="OPTIMIZATION-TRAINING-HELD-OUT-SEPARATE"
#trainExecutionMode = "OPTIMIZATION-TRAINING-CROSS-VALIDATION"

qsub -q hoffmangroup -N prediction_${label} -cwd -b y -o $outputFileName -e $outputFileName th multiple_siamese_nn_toy.lua $chromSel $folderName $dataSource $trainSamplesPerc $trainNegElemsPerc $trainTupleLimit $valNegElemsPerc $valTupleLimit $trainExecutionMode



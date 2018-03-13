#!/bin/bash
#
#$ -cwd
#$ -S /bin/bash
#
set -o nounset -o pipefail -o errexit
set -o xtrace

label="2016-05-03_allChromosomes"
chromSel="chrY"

randomNumber=$(shuf -i1-10000 -n1)
# folderName="../results/${label}_tests_${chromSel}"
folderName="../results/${label}_tests_CrossVali_allChroms"
# mkdir $folderName
outputFileName="$folderName/multiple_output_${chromSel}_rand${randomNumber}"
echo $outputFileName


dataSource="Thurman_Miriam"
trainSamplesPerc=50
trainNegElemsPerc=50
trainTupleLimit=5000

valNegElemsPerc=99
valTupleLimit=5000

# trainExecutionMode="OPTIMIZATION-TRAINING-CROSS-VALIDATION"
# trainExecutionMode="OPTIMIZATION-TRAINING-HELD-OUT-DISTAL"
# trainExecutionMode="OPTIMIZATION-TRAINING-HELD-OUT"

# trainExecutionMode="OPTIMIZATION-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT"
# trainExecutionMode="OPTIMIZATION-TRAINING-CROSS-VALIDATION-DOUBLE-INPUT"

trainExecutionMode="SINGLE-MODEL-TRAINING-CROSS-VALIDATION"

qsub -q hoffmangroup -N prediction_${label} -cwd -b y -o $outputFileName -e $outputFileName th multiple_siamese_nn_toy.lua $chromSel $folderName $dataSource $trainSamplesPerc $trainNegElemsPerc $trainTupleLimit $valNegElemsPerc $valTupleLimit $trainExecutionMode



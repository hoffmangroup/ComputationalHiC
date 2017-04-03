#!/bin/bash
#
#$ -cwd
#$ -S /bin/bash
#
set -o nounset -o pipefail -o errexit
set -o xtrace

indexStartVector=(10140 10280 60240 16620 10380 147760 10080 21080 10000 81700 76920 62180 19062120 19058540 20049440 60040 140 10040 89240 60340 9418900 16050580 2699320)

indexEndVector=(249240470 243185010 197947210 191021710 180900710 171048270 159128570 146301350 141111390 135524710 134946470 133841550 115109750 107287650 102521110 90278910 81195070 78016350 59118830 62964870 48111930 51238390 154917490)


dnaseColNumToExclude=-1
profi_flag="false"

## IMR90 -> 57
## HUVEC -> 55
## k562 -> 61
## GM12878 -> 26 (GM12865)

#trainingSetCellTypeName="-1"
trainingSetCellTypeName="HUVEC"

#hicCellTypeNameToHighlight="IMR90"
#hicCellTypeNumberToHighlight=57

hicCellTypeNameToHighlight="GM12878"
hicCellTypeNumberToHighlight=26

# hicCellTypeNameToHighlight="k562"
# hicCellTypeNumberToHighlight=61

# hicCellTypeNameToHighlight="IMR90"
# hicCellTypeNumberToHighlight=57

#hicCellTypeNameToHighlight="HUVEC"
#hicCellTypeNumberToHighlight=55

today=`date +%Y-%m-%d`

# OPTIMIZATION-TRAINING-HELD-OUT-DISTAL
execution="SINGLE-MODEL-TRAINING-HELD-OUT-DISTAL"

#tupleLimit=20000

tupleLimit=30000
proportion=1000
mem_size=$((tupleLimit / proportion))
#mem_size=48


folder="../results/"$today"_optim_allChrom_regExc_train_on_"${trainingSetCellTypeName}"_test_on_"${hicCellTypeNameToHighlight}"_"${tupleLimit}"elems"/
mkdir -p $folder

i=1
for i in $(seq $((${#indexStartVector[@]} - 1)))
do

#   if [ "$i" -ne 1 ] && [ "$i" -ne 2 ]  && [ "$i" -ne 3 ]  && [ "$i" -ne 7 ]  && [ "$i" -ne 12 ]  && [ "$i" -ne 14 ] && [ "$i" -ne 16 ] && [ "$i" -ne 19 ] && [ "$i" -ne 20 ] 
#   then 
   
   trainStart=${indexStartVector[$i-1]}
   trainEnd=${indexEndVector[$i-1]}

   chrNum="chr"${i}
   random_number=$(shuf -i1-100000 -n1)
  
  outputFile=${folder}${chrNum}_train_complete-${trainStart}-${trainEnd}-yesMinibatch_${tupleLimit}elems_bal_${random_number}rand
  
  modelFile="./models/${chrNum}_"${hicCellTypeNameToHighlight}"_whole_chromosome__trained_model_yesMinibatch_${tupleLimit}elems_bal_${random_number}rand"

  qsub -q hoffmangroup  -l mem_requested=${mem_size}G -N ${trainingSetCellTypeName}_${hicCellTypeNameToHighlight}_${chrNum}_pred${tupleLimit} -cwd -b y -o $outputFile -e $outputFile th siamese_nn_toy.lua  prediction  $tupleLimit  $chrNum  $trainStart  $trainEnd  50  -1  $outputFile  $execution  $modelFile  false  $trainStart  $trainEnd  2000  90  -1  -1  true  20 $dnaseColNumToExclude $hicCellTypeNameToHighlight $hicCellTypeNumberToHighlight $profi_flag $trainingSetCellTypeName > $outputFile 2> $outputFile


done

i=23
chrNum="chrX"
trainStart=${indexStartVector[$i-1]}
trainEnd=${indexEndVector[$i-1]}  

random_number=$(shuf -i1-100000 -n1)

outputFile=${folder}${chrNum}_train_complete-${trainStart}-${trainEnd}-yesMinibatch_${tupleLimit}elems_bal_${random_number}rand
  
modelFile="./models/${chrNum}_"${hicCellTypeNameToHighlight}"_whole_chromosome__trained_model_yesMinibatch_${tupleLimit}elems_bal_${random_number}rand"

qsub -q hoffmangroup  -l mem_requested=${mem_size}G -N ${trainingSetCellTypeName}_${hicCellTypeNameToHighlight}_${chrNum}_pred${tupleLimit} -cwd -b y -o $outputFile -e $outputFile th siamese_nn_toy.lua  prediction  $tupleLimit  $chrNum  $trainStart  $trainEnd  50  -1  $outputFile  $execution  $modelFile  false  $trainStart  $trainEnd  2000  90  -1  -1  true  20 $dnaseColNumToExclude $hicCellTypeNameToHighlight $hicCellTypeNumberToHighlight $profi_flag $trainingSetCellTypeName > $outputFile 2> $outputFile

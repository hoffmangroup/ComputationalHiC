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

trainingSetCellTypeName1="GM12878"
trainingSetCellTypeName2="HUVEC"
trainingSetCellTypeName3="IMR90"
trainingSetCellTypeName4="-1"

hicCellTypeNameValidSet1="k562"
hicCellTypeNameValidSet2="-1"
hicCellTypeNameValidSet3="-1"
hicCellTypeNameValidSet4="-1"


today=`date +%Y-%m-%d`

# OPTIMIZATION-TRAINING-HELD-OUT-DISTAL
execution="SINGLE-MODEL-TRAINING-HELD-OUT-DISTAL"

#tupleLimit=20000

tupleLimit=20000
proportion=1000
mem_size=$((tupleLimit / proportion))
#mem_size=48

testTupleLimit=2000
balancedFalsePerc=50
training_sample_perc=-1
retrieveFP_flag_ini="false"

folder="../results/"$today"_multi_cell_test_"${hicCellTypeNameValidSet1}"_"${tupleLimit}"elems"/
mkdir -p $folder

i=1
for i in $(seq $((${#indexStartVector[@]} - 1)))
do


#   if [ "$i" -ne 1 ] && [ "$i" -ne 2 ]  && [ "$i" -ne 3 ]  && [ "$i" -ne 7 ]  && [ "$i" -ne 12 ]  && [ "$i" -ne 14 ] && [ "$i" -ne 16 ] && [ "$i" -ne 19 ] && [ "$i" -ne 20 ] 
#   then 
   
   trainStart=${indexStartVector[$i-1]}
   trainEnd=${indexEndVector[$i-1]}

   chrNum="chr"${i}
   # chrNum="chr0"
   random_number=$(shuf -i1-100000 -n1)
  
  outputFile=${folder}${chrNum}_train_complete-${trainStart}-${trainEnd}_test_${hicCellTypeNameValidSet1}_${tupleLimit}elems_bal_${random_number}rand
  
  modelFile="./models/${chrNum}_trained_model_"${hicCellTypeNameValidSet1}"_${tupleLimit}elems_bal_${random_number}rand"

#   qsub -q hoffmangroup  -l mem_requested=${mem_size}G -N ${chrNum}_job${random_number} -cwd -b y -o $outputFile -e $outputFile th siamese_nn_toy.lua  prediction  $tupleLimit  $chrNum  $trainStart  $trainEnd  50  -1  $outputFile  $execution  $modelFile  false  $trainStart  $trainEnd  2000  90  -1  -1  true  20 $dnaseColNumToExclude $hicCellTypeNameValidSet1 $hicCellTypeNumberToHighlight $profi_flag $trainingSetCellTypeName1 > $outputFile 2> $outputFile

  echo -q hoffmangroup  -l mem_requested=${mem_size}G -N ${chrNum}_job${random_number} -cwd -b y -o $outputFile -e $outputFile th siamese_nn_toy.lua prediction $tupleLimit  $chrNum  $trainStart  $trainEnd  $balancedFalsePerc  $training_sample_perc  $outputFile  $execution  $modelFile  $retrieveFP_flag_ini $trainStart  $trainEnd $testTupleLimit 90 -1 -1 true 20 -1 $hicCellTypeNameValidSet1 $hicCellTypeNameValidSet2 $hicCellTypeNameValidSet3 $hicCellTypeNameValidSet4 $profi_flag $trainingSetCellTypeName1 $trainingSetCellTypeName2 $trainingSetCellTypeName3 $trainingSetCellTypeName4 > $outputFile 2> $outputFile


done

i=23
chrNum="chrX"
trainStart=${indexStartVector[$i-1]}
trainEnd=${indexEndVector[$i-1]}  

random_number=$(shuf -i1-100000 -n1)

outputFile=${folder}${chrNum}_train_complete-${trainStart}-${trainEnd}_test_${hicCellTypeNameValidSet1}_${tupleLimit}elems_bal_${random_number}rand
  
modelFile="./models/${chrNum}_trained_model_"${hicCellTypeNameValidSet1}"_${tupleLimit}elems_bal_${random_number}rand"


echo -q hoffmangroup  -l mem_requested=${mem_size}G -N ${chrNum}_job${random_number} -cwd -b y -o $outputFile -e $outputFile th siamese_nn_toy.lua prediction $tupleLimit  $chrNum  $trainStart  $trainEnd  $balancedFalsePerc  $training_sample_perc  $outputFile  $execution  $modelFile  $retrieveFP_flag_ini  $trainStart  $trainEnd $testTupleLimit 90 -1 -1 true 20 -1 $hicCellTypeNameValidSet1 $hicCellTypeNameValidSet2 $hicCellTypeNameValidSet3 $hicCellTypeNameValidSet4 $profi_flag $trainingSetCellTypeName1 $trainingSetCellTypeName2 $trainingSetCellTypeName3 $trainingSetCellTypeName4 > $outputFile 2> $outputFile

# # on all the chromosomes
# 
# chrNum="chr0"
# trainStart=${indexStartVector[$i-1]}
# trainEnd=${indexEndVector[$i-1]}  
# 
# random_number=$(shuf -i1-100000 -n1)
# 
# outputFile=${folder}${chrNum}_train_complete-${trainStart}-${trainEnd}_test_${hicCellTypeNameValidSet1}_${tupleLimit}elems_bal_${random_number}rand
#   
# modelFile="./models/${chrNum}_trained_model_"${hicCellTypeNameValidSet1}"_${tupleLimit}elems_bal_${random_number}rand"
# 
# 
# qsub -q hoffmangroup  -l mem_requested=${mem_size}G -N ${chrNum}_job${random_number} -cwd -b y -o $outputFile -e $outputFile th siamese_nn_toy.lua prediction $tupleLimit  $chrNum  $trainStart  $trainEnd  $balancedFalsePerc  $training_sample_perc  $outputFile  $execution  $modelFile  $retrieveFP_flag_ini  $trainStart  $trainEnd $testTupleLimit 90 -1 -1 true 20 -1 $hicCellTypeNameValidSet1 $hicCellTypeNameValidSet2 $hicCellTypeNameValidSet3 $hicCellTypeNameValidSet4 $profi_flag $trainingSetCellTypeName1 $trainingSetCellTypeName2 $trainingSetCellTypeName3 $trainingSetCellTypeName4 > $outputFile 2> $outputFile

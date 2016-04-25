require "../../_project/bin/database_management.lua"
require "../../_project/bin/utils.lua"

print('\n\n @ @ @ @ @ @ START @ @ @ @ @ @ @ ');
print('file: multiple_siamese_nn_toy.lua');
print('author Davide Chicco <davide.chicco@gmail.com>');
print(os.date("%c", os.time()));

TEST_FLAG = true

MAX_ARGS_NUM = 8

local chromSel = tostring(arg[1]);
local folderName = tostring(arg[2]);
-- local folderName = "../results/tests_"..tostring(chromSel).."/"

local dataSource = tostring(arg[3]); -- "Thurman_Miriam"
local trainSamplesPerc = tonumber(arg[4]); -- 90, if not SEPARATE

local trainNegElemsPerc = tonumber(arg[5]); -- 90
local trainTupleLimit = tonumber(arg[6]); -- 5000
local true_interactions_spanSize = trainTupleLimit*1.5 -- to have enough room

local val_negElemsPerc = tonumber(arg[7]); -- 90
local val_tupleLimit = tonumber(arg[8]); -- 5000

-- local trainExecutionMode = "OPTIMIZATION-TRAINING-HELD-OUT-SEPARATE";
-- local trainExecutionMode = "OPTIMIZATION-TRAINING-CROSS-VALIDATION";
local trainExecutionMode = tostring(arg[9]);

io.write("==> command: th multiple_siamese_nn_toy.lua ")
for i=1,MAX_ARGS_NUM do
  io.write(arg[i].." ")
end
io.write("\n");

if trainNegElemsPerc >= 100 then print("Error: trainNegElemsPerc must be < 100, while it is "..trainNegElemsPerc.."%") end
if val_negElemsPerc >= 100 then print("Error: val_negElemsPerc must be < 100, while it is "..val_negElemsPerc.."%") end


-- local mkdirCommand = " mkdir "..folderName;
-- execute_command(mkdirCommand);
-- print("Command launched:\t\t\t".. mkdirCommand);



local vect = selectGenomeSpanIndices_bySpanSize(chromSel, true_interactions_spanSize, dataSource);

local indices_start = vect[1];
local indices_end = vect[2];
local modelFileName = "";

for i=1,#indices_start do
  
  if i==1 then 

    local chrStart = indices_start[i];
    local chrEnd = indices_end[i];
    local region = chromSel.."-"..tostring(chrStart).."-"..tostring(chrEnd);
    
    local label = "prediction_"..tostring(os.time()).."time_"..chromSel.."-"..tostring(chrStart).."-"..tostring(chrEnd).."_trainPer"..trainSamplesPerc.."_negPer"..tostring(trainNegElemsPerc);
    label = label.."_trainTuples"..tostring(trainTupleLimit);
    outputFile=folderName.."/"..label.."_VALID"
    
    
     local val_chrStart_locus = tostring(indices_start[i+1]);
     local val_chrEnd_locus = tostring(indices_end[i+1]);
     local val_tuple_limit = tostring(val_tupleLimit); -- TO BE CHANGED
     local val_balancedFalsePerc = tostring(val_negElemsPerc); -- TO BE CHANGED
    
    modelFileName = "./models/"..region.."_"..tostring(os.time()).."time_trained_model";
    local retriveFP_flag = false;

    local command = "qsub -q hoffmangroup -N "..label.." -cwd -b y -o "..outputFile.." -e "..outputFile.." th siamese_nn_toy.lua "..label.." "..trainTupleLimit.." "..chromSel.." "..chrStart.." "..chrEnd.." "..trainNegElemsPerc.." "..trainSamplesPerc.." "..outputFile .." "..tostring(trainExecutionMode).." "..tostring(modelFileName).." "..tostring(retriveFP_flag);
    
    command = command.." "..val_chrStart_locus.." "..val_chrEnd_locus.." "..val_tuple_limit.." "..val_balancedFalsePerc;

    execute_command(command);
    print("Command launched:\t\t\t".. command);
    
    if TEST_FLAG == true then
      file_present = wait_until_the_file_exists(modelFileName);
    end
  
  elseif i>=3 and file_present==true then
    
    
    local chrStart = indices_start[i];
    local chrEnd = indices_end[i];
    local region = chromSel.."-"..tostring(chrStart).."-"..tostring(chrEnd);
    
    local label = "prediction_"..tostring(os.time()).."time_"..chromSel.."-"..tostring(chrStart).."-"..tostring(chrEnd).."_TEST";
    outputFile=folderName.."/"..label
    
    local executionMode = "JUST-TESTING"
    local retriveFP_flag = false;
    trainTupleLimit = -1;

    local command = "qsub -q hoffmangroup -N "..label.." -cwd -b y -o "..outputFile.." -e "..outputFile.." th siamese_nn_toy.lua "..label.." "..trainTupleLimit.." "..chromSel.." "..chrStart.." "..chrEnd.." "..trainNegElemsPerc.." "..trainSamplesPerc.." "..outputFile .." "..tostring(executionMode).." "..tostring(modelFileName).." "..tostring(retriveFP_flag);
    command = command.." -1 -1 -1 -1";

    if TEST_FLAG == true then
     execute_command(command);
     print("Command launched:\t\t\t".. command);
    end
    
    
    
  end

end


print('file: multiple_siamese_nn_toy.lua');
print('\n\n @ @ @ @ @ @ END @ @ @ @ @ @ @ ');


--  th siamese_nn_toy.lua $label -1  $chromSel $chrStart_locus $chrEnd_locus $trainNegElemsPerc $trainSamplesPerc $outputFile "JUST-TESTING" ./models/1460409441_model_chr20-62168520-62409970 false
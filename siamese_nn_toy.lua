
-- number=$(( ( RANDOM % 100000 )  + 1 ))
-- tupleLimit=2000
-- chromSel="chr21"
-- chrStart_locus=46721100
-- chrEnd_locus=47409790	
-- negativeElementsPerc=80
-- trainSamplesPerc=80
-- label="newTest"
-- ######## Line too long (146 chars) ######## :
-- outputFile=./tests/$label\_$chromSel-$chrStart_locus-$chrEnd_locus\_negPerc$negativeElementsPerc\_trainPerc$trainSamplesPerc\_tuples$tupleLimit
-- ######## Line too long (218 chars) ######## :
-- qsub -q hoffmangroup -N siamese_nn_toy -cwd -b y -o ./$outputFile -e ./$outputFile th siamese_nn_toy.lua $label $tupleLimit $chromSel $chrStart_locus $chrEnd_locus $negativeElementsPerc $trainSamplesPerc $outputFile



-- SAVE_MSE_VECTOR_TO_FILE = true

L2_WEIGHT = 0.000001;
L1_WEIGHT = 0.005;
REGULARIZATION = true;
MINIBATCH = false;
NEW_OPTIM_GRADIENT_MINIBATCH = true; -- to set better

PERMUTATION_TRAIN = true;
PERMUTATION_TEST = false;
-- MINIBATCH_SPAN_NUMBER = 10
K_FOLD = 5;

SUFFICIENT_ACCURACY = 0.9;
SUFFICIENT_MCC = 0.9;

XAVIER_INITIALIZATION = false;
MOMENTUM_FLAG = true;
MOMENTUM_ALPHA = 0.5;
PRINT_NOT_REG_CELL_TYPE_ONCE = false;
ITERATIONS_CONST = 1000; -- ******** 1000 **********
LEARNING_RATE_CONST = 0.001; -- ******** 0.001 **********
MAX_POSSIBLE_MSE = 4;
CELL_TYPE_NUMBER = 82;
READ_DATA_FROM_DB = true; -- ********* true *********
NO_INTERSECTION_BETWEEN_SETS = true; -- ******** true **********
PRINT_NOT_REG_INDICES_ONCE = true
PRINT_NOT_REG_CELL_TYPE_ONCE = true
SCORE_UNDEF = -2
HIGHLIGHT_NEURON_WEIGHT_1ST_CELL_TYPE = false

require 'optim'
-- require '../../torch/NEW_CosineDistance.lua'
require './lib/NEW_CosineDistance.lua'

local globalArrayFPindices = {}
local globalArrayFPvalues = {}
globalMinFPplusFN_vector = {}

-- Number of the cell type in the DNase cell type list
GM12878_dnaseCellType = 26
HUVEC_dnaseCellType = 55
IMR90_dnaseCellType = 57
k562_dnaseCellType = 61

-- function that retrieves the column name and number from the cell type
function retrieveCellTypeColumnNameAndNumber(thisHicCellTypeSpecific)
 if thisHicCellTypeSpecific~="-1"
 and thisHicCellTypeSpecific~=-1
 and thisHicCellTypeSpecific~="GM12878"
 and thisHicCellTypeSpecific~="HMEC"
 and thisHicCellTypeSpecific~="HUVEC"
 and thisHicCellTypeSpecific~="HeLa"
 and thisHicCellTypeSpecific~="IMR90"
 and thisHicCellTypeSpecific~="k562"
 and thisHicCellTypeSpecific~="KBM7"
 and thisHicCellTypeSpecific~="NHEK" then
  
 io.write("Error: the thisHicCellTypeSpecific = "..thisHicCellTypeSpecific.." is not one ")
 io.write(" of the 8 cell type available in the Hi-C dataset (GM12878, HMEC, HUVEC, ")
 io.write(" HeLa, IMR90, k562, KBM7, NHEK). The program will stop")
 io.flush()
 os.exit()  
 end

 local dnaseCellTypeToHighlightNumber = -1

 if thisHicCellTypeSpecific=="GM12878" then dnaseCellTypeToHighlightNumber = GM12878_dnaseCellType end
 if thisHicCellTypeSpecific=="HUVEC" then dnaseCellTypeToHighlightNumber = HUVEC_dnaseCellType end
 if thisHicCellTypeSpecific=="IMR90" then dnaseCellTypeToHighlightNumber = IMR90_dnaseCellType end
 if thisHicCellTypeSpecific=="k562" then dnaseCellTypeToHighlightNumber = k562_dnaseCellType end


 local dnaseCellTypeToHighlightName = "";
 if dnaseCellTypeToHighlightNumber>=1 and dnaseCellTypeToHighlightNumber<=CELL_TYPE_NUMBER then
  
  local columnNames = getColumnNamesOfTable("chromregionprofiles")
  local dnaseExcludeColumnName = columnNames[dnaseExcludeColumn]
  
  dnaseCellTypeToHighlightName = columnNames[dnaseCellTypeToHighlightNumber]
  print("RETRIEVING THE FEATURE-COLUMN "..dnaseCellTypeToHighlightName.." number "..dnaseCellTypeToHighlightNumber.." among "..CELL_TYPE_NUMBER);
 elseif dnaseCellTypeToHighlightNumber==-1 or dnaseCellTypeToHighlightNumber=="-1" then
  print("No cell type will be retrieved in the input DNase table");
 else
  print("Error: the dnaseCellTypeToHighlightNumber = "..dnaseCellTypeToHighlightNumber.." is not in the 1- "..CELL_TYPE_NUMBER.." interval. The program will stop");
  os.exit();  
 end
 
 return dnaseCellTypeToHighlightNumber, dnaseCellTypeToHighlightName;
end

--
-- Function that does not apply the regularization only to the three 

-- This function does NOT apply the regularization to the feature indEX of the  cell type INDICATED
-- 
-- new version of sgd_NEW
--
-- - hidden_units = number of hidden units of the neural network
-- - input_number = number of input units of the neural network
-- The rest is the same of sgd.
-- ######## Line too long (106 chars) ######## :
function optim.sgd_1cellTypeHighlight(opfunc, x, config, state, feature_index, hidden_units, input_number)

-- ######## Line too long (116 chars) ######## :
-- io.write("feature_index = "..feature_index.."\thidden_units = "..hidden_units.."\tinput_number = "..input_number)
io.flush()

 -- (0) get/update state
 local config = config or {}
 local state = state or config
 local lr = config.learningRate or 1e-3
 local lrd = config.learningRateDecay or 0
 local wd = config.weightDecay or 0
 local mom = config.momentum or 0
 local damp = config.dampening or mom
 local nesterov = config.nesterov or false
 local lrs = config.learningRates
 local wds = config.weightDecays
 state.evalCounter = state.evalCounter or 0
 local nevals = state.evalCounter
-- ######## Line too long (108 chars) ######## :
 assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

 -- (1) evaluate f(x) and df/dx
 local fx,dfdx = opfunc(x)
 
 -- print("hidden_units = "..hidden_units);
 -- print("input_number = "..input_number);
 
 -- new part
 local original_dfdx = dfdx:clone()

 -- (2) weight decay with single or individual parameters
 if wd ~= 0 then
 
  dfdx:add(wd, x)
  
   local idp = 5
-- ######## Line too long (82 chars) ######## :
   if PRINT_NOT_REG_INDICES_ONCE then  io.write("not-regularized indices = "); end
   for i=0,hidden_units-1 do
    -- io.write("(i="..i..") ")
    --io.write("index = "..feature_index.." +("..i.."*"..input_number..")");
    io.flush();
    local index = feature_index+(i*input_number) 
    
    
    dfdx[index] = original_dfdx[index]
    if PRINT_NOT_REG_INDICES_ONCE then 
    io.write("[index="..index.."]\t")
    io.flush()
    end
   end
   
   if PRINT_NOT_REG_INDICES_ONCE then 
    
    io.write("\n")
    io.flush()
   end
   


 elseif wds then
  if not state.decayParameters then
   state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
  end
  state.decayParameters:copy(wds):cmul(x)
  dfdx:add(state.decayParameters)
 end
 
 PRINT_NOT_REG_INDICES_ONCE = false
 -- (3) apply momentum
 if mom ~= 0 then
  if not state.dfdx then
   state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
  else
   state.dfdx:mul(mom):add(1-damp, dfdx)
  end
  if nesterov then
   dfdx:add(mom, state.dfdx)
  else
   dfdx = state.dfdx
  end
 end

 -- (4) learning rate decay (annealing)
 local clr = lr / (1 + nevals*lrd)
 
 -- (5) parameter update with single or individual learning rates
 if lrs then
  if not state.deltaParameters then
   state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
  end
  state.deltaParameters:copy(lrs):cmul(dfdx)
  x:add(-clr, state.deltaParameters)
 else
  x:add(-clr, dfdx)
 end

 -- (6) update evaluation counter
 state.evalCounter = state.evalCounter + 1

 
 -- return x*, f(x) before optimization
 return x,{fx}
end
 


--
-- Function that does not apply the regularization only to the three 

-- ######## Line too long (168 chars) ######## :
-- This function does NOT apply the regularization to the feature indices of the four cell types shared by DNase dataset and Hi-C dataset, except the feature_index one:
-- GM12878_dnaseCellType = 26
-- HUVEC_dnaseCellType = 55
-- IMR90_dnaseCellType = 57
-- k562_dnaseCellType = 61
-- 
-- 
-- - hidden_units = number of hidden units of the neural network
-- - input_number = number of input units of the neural network
-- The rest is the same of sgd.
-- ######## Line too long (107 chars) ######## :
function optim.sgd_3cellTypesHighlight(opfunc, x, config, state, feature_index, hidden_units, input_number)

-- ######## Line too long (116 chars) ######## :
-- io.write("feature_index = "..feature_index.."\thidden_units = "..hidden_units.."\tinput_number = "..input_number)
io.flush()

 -- (0) get/update state
 local config = config or {}
 local state = state or config
 local lr = config.learningRate or 1e-3
 local lrd = config.learningRateDecay or 0
 local wd = config.weightDecay or 0
 local mom = config.momentum or 0
 local damp = config.dampening or mom
 local nesterov = config.nesterov or false
 local lrs = config.learningRates
 local wds = config.weightDecays
 state.evalCounter = state.evalCounter or 0
 local nevals = state.evalCounter
-- ######## Line too long (108 chars) ######## :
 assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

 -- (1) evaluate f(x) and df/dx
 local fx,dfdx = opfunc(x)
 
 -- print("hidden_units = "..hidden_units);
 -- print("input_number = "..input_number);
 
 -- new part
 local original_dfdx = dfdx:clone()

 -- (2) weight decay with single or individual parameters
 if wd ~= 0 then
 
  dfdx:add(wd, x)
  
   local idp = 5
-- ######## Line too long (82 chars) ######## :
   if PRINT_NOT_REG_INDICES_ONCE then  io.write("not-regularized indices = "); end
   for i=0,hidden_units-1 do
    -- io.write("(i="..i..") ")
    local index_GM12878 = GM12878_dnaseCellType+(i*input_number) 
    local index_HUVEC = HUVEC_dnaseCellType+(i*input_number)
    local index_IMR90 = IMR90_dnaseCellType+(i*input_number)
    local index_k562 = k562_dnaseCellType+(i*input_number)
    if PRINT_NOT_REG_INDICES_ONCE then 
    
-- ######## Line too long (95 chars) ######## :
    if feature_index~=GM12878_dnaseCellType then io.write("["..index_GM12878.."]\t") end	      
-- ######## Line too long (91 chars) ######## :
    if feature_index~=HUVEC_dnaseCellType then io.write("["..index_HUVEC.."]\t") end	      
-- ######## Line too long (91 chars) ######## :
    if feature_index~=IMR90_dnaseCellType then io.write("["..index_IMR90.."]\t") end	      
-- ######## Line too long (82 chars) ######## :
    if feature_index~=k562_dnaseCellType then io.write("["..index_k562.."]\t") end
    io.write("\n")
    io.flush();
    if i%15==0 then io.write("\n"); end
 
    end
    
-- ######## Line too long (110 chars) ######## :
    if feature_index~=GM12878_dnaseCellType then dfdx[index_GM12878] = original_dfdx[index_GM12878] end	      
-- ######## Line too long (104 chars) ######## :
    if feature_index~=HUVEC_dnaseCellType then dfdx[index_HUVEC] = original_dfdx[index_HUVEC] end	      
-- ######## Line too long (104 chars) ######## :
    if feature_index~=IMR90_dnaseCellType then dfdx[index_IMR90] = original_dfdx[index_IMR90] end	      
-- ######## Line too long (94 chars) ######## :
    if feature_index~=k562_dnaseCellType then dfdx[index_k562] = original_dfdx[index_k562] end
    
   end
   
   if PRINT_NOT_REG_INDICES_ONCE then 
    io.write("\n")
    io.flush()
   end
   
   PRINT_NOT_REG_INDICES_ONCE = false

 elseif wds then
  if not state.decayParameters then
   state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
  end
  state.decayParameters:copy(wds):cmul(x)
  dfdx:add(state.decayParameters)
 end

 -- (3) apply momentum
 if mom ~= 0 then
  if not state.dfdx then
   state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
  else
   state.dfdx:mul(mom):add(1-damp, dfdx)
  end
  if nesterov then
   dfdx:add(mom, state.dfdx)
  else
   dfdx = state.dfdx
  end
 end

 -- (4) learning rate decay (annealing)
 local clr = lr / (1 + nevals*lrd)
 
 -- (5) parameter update with single or individual learning rates
 if lrs then
  if not state.deltaParameters then
   state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
  end
  state.deltaParameters:copy(lrs):cmul(dfdx)
  x:add(-clr, state.deltaParameters)
 else
  x:add(-clr, dfdx)
 end

 -- (6) update evaluation counter
 state.evalCounter = state.evalCounter + 1

 
 -- return x*, f(x) before optimization
 return x,{fx}
end
 


-- Function siameseDistanceApplication()
function siameseDistanceApplication(current_dataset)

  local resultVector = {} 
  local cosineFun = nn.CosineDistance(); 
  local lun = (current_dataset[1]):size()[1]

  for i=1,lun do
  -- ######## Line too long (102 chars) ######## :
  -- resultVector[#resultVector+1] = myCosineApplication(current_dataset[1][i], current_dataset[2][i])  
  -- ######## Line too long (100 chars) ######## :
  resultVector[#resultVector+1] = cosineFun:forward({current_dataset[1][i], current_dataset[2][i]})[1]
  end

  return torch.Tensor(resultVector);
end


-- Function myCosineApplication()
function myCosineApplication(tensorA, tensorB)

  local result = SCORE_UNDEF;

  local A_size = tensorA:size()[1];
  local B_size = tensorB:size()[1];

  if tensorA:size()[1] ~= tensorA:size()[1] then 
  print("Impossible to compute the cosine similarity")
  end
  
  local sup = 0;
  local A_sum_square = 0;
  local B_sum_square = 0;
  for i=1,A_size do
  sup = sup + (tensorA[i]*tensorB[i]);
  A_sum_square = A_sum_square + (tensorA[i]*tensorA[i]);
  B_sum_square = B_sum_square + (tensorB[i]*tensorB[i]);
  end  

  local A_sum_square_root = math.sqrt(A_sum_square);
  local B_sum_square_root = math.sqrt(B_sum_square);
  result = sup / (A_sum_square_root * B_sum_square_root);

  return result;
end

-- ######## Line too long (84 chars) ######## :
-- Function that takes the training set and arrange them for the minibatch splitting
-- ######## Line too long (93 chars) ######## :
function arrangeSetsForMinibatch(first_datasetTrain, second_datasetTrain, targetDatasetTrain)

-- io.write("arrangeSetsForMinibatch() start :: ");

 local printPercCount = 0;
 local trainDataset = {};
 local targetDataset = {};

 local leftVect = {}
 local rightVect = {}
   
 local normTargetDatasetTrain = {}
 local rate = -1
 for q=1,#first_datasetTrain do
 
 rate = round(q*100/#first_datasetTrain,2);
 -- if ((rate*10)%10==0) then io.write(rate.."% "); end
 
 trainDataset[q]={first_datasetTrain[q], second_datasetTrain[q]}	  
 normTargetDatasetTrain[q] = (targetDatasetTrain[q][1]*2) - 1

 -- no Sequencer
 leftVect[q] = trainDataset[q][1]; 
 rightVect[q] = trainDataset[q][2];
 
 if q==1 then       
  leftTens = leftVect[q]; rightTens =rightVect[q];
 else       
  leftTens = torch.cat(leftTens, leftVect[q], 2);
  rightTens = torch.cat(rightTens, rightVect[q], 2);       
 end
 
 collectgarbage();  
 end
 
 local newleftTens = leftTens:transpose(1,2); 
 local newrightTens = rightTens:transpose(1,2);

-- io.write(" :: arrangeSetsForMinibatch() end");

return {newleftTens, newrightTens, normTargetDatasetTrain};
end

-- Function that creates the minibatch sets
function createMinibatch(minibatchSize, m, newleftTens, newrightTens, normTargetDatasetTrain)

  local minibatch_train_c = torch.Tensor(minibatchSize)
  local target_train_c = torch.Tensor(minibatchSize)

  local lower_index = 1+minibatchSize*(m-1)	
  local upper_index = -1;
  
  if m~=MINIBATCH_SPAN_NUMBER then
  upper_index = (m-1)*minibatchSize+minibatchSize
  else
  upper_index = #normTargetDatasetTrain
  end	
  
  local temp_train_left = newleftTens[{{lower_index,upper_index}}]
  local temp_train_right = newrightTens[{{lower_index,upper_index}}]	
  minibatch_train_c = {temp_train_left, temp_train_right}	
  target_train_c = subtable(normTargetDatasetTrain, lower_index, upper_index)

  return {minibatch_train_c, target_train_c};
end

-- Function that returns the majority between three elements
function majorityGreaterThanThreshold(value1, value2, value3, threshold)

-- ######## Line too long (90 chars) ######## :
if value1 >= threshold and (value2 >= threshold or value3 >= threshold) then return 1; end
-- ######## Line too long (90 chars) ######## :
if value2 >= threshold and (value1 >= threshold or value3 >= threshold) then return 1; end
-- ######## Line too long (90 chars) ######## :
if value3 >= threshold and (value1 >= threshold or value2 >= threshold) then return 1; end

return 0;
end

-- Function that runs the k fold cross validation
function kfold_cross_validation(dataset_firstChromRegion, dataset_secondChromRegion, targetVector, initialPerceptron, architectureLabel, DATA_SIZE)

  local span_size = math.floor(DATA_SIZE/K_FOLD);
  local globalPredictionVector = {}
  local truthVector = {}

   -- START K_FOLD CROSS VALIDATION LOOP -- 
  for k=1,K_FOLD do
  
  local test_initial_index = -1
  local test_last_index = -1
  
  if k==1 then	   
    test_initial_index = 1
    test_last_index = span_size
  elseif k==K_FOLD then
    test_initial_index = (span_size*(k-1))+1
    test_last_index = DATA_SIZE
  else
    test_initial_index = (span_size*(k-1))+1
    test_last_index = span_size*(k)	      
  end
  
-- ######## Line too long (173 chars) ######## :
  io.write("\n\n * * * * k-fold cross validation \t k = "..k.."/"..K_FOLD)
  io.write(" test_initial_index = "..test_initial_index.."  test_last_index = "..test_last_index.. " * * * * ")
  io.flush()
  
  local first_datasetTrain = {}
  local second_datasetTrain = {}
  local targetDatasetTrain = {}
  
  local first_datasetTest  = {}
  local second_datasetTest = {}
  local targetDatasetTest = {}
  
  for i=1,DATA_SIZE do	   
   if i >= test_initial_index and i <= test_last_index then
   first_datasetTest[#first_datasetTest+1]  = dataset_firstChromRegion[i];
   second_datasetTest[#second_datasetTest+1] = dataset_secondChromRegion[i];
   targetDatasetTest[#targetDatasetTest+1] = targetVector[i];
   else
   first_datasetTrain[#first_datasetTrain+1]  = dataset_firstChromRegion[i];
   second_datasetTrain[#second_datasetTrain+1] = dataset_secondChromRegion[i];
   targetDatasetTrain[#targetDatasetTrain+1] = targetVector[i];	     
   end	   
  end
  	    
  local applicationOutput = siameseNeuralNetwork_application(first_datasetTrain, second_datasetTrain, targetDatasetTrain, first_datasetTest, second_datasetTest, targetDatasetTest, initialPerceptron, architectureLabel, trainedModelFile);
    
  local lastAccuracy = applicationOutput[1];
  local predictionVect = applicationOutput[2];	
  local lastMCC = applicationOutput[3];	
  local sortedTruthVect = applicationOutput[4];	
  
  local c=1;
  for t=test_initial_index,test_last_index do
   globalPredictionVector[t]=predictionVect[c];
   truthVector[t]=sortedTruthVect[c];
   c = c+1;
  end
  
  end 
  -- END K_FOLD CROSS VALIDATION LOOP-- 
  
  return {globalPredictionVector, truthVector};
end



-- function that checks the stop condition for each iteration
function continueOrStopCheck(lastAccuracy, lastMCC)
  
  local stopConditionFlag = false;  
  print("lastMCC = "..lastMCC);
 
  if lastMCC >= SUFFICIENT_MCC then 
   
   saveModelToFile(generalPerceptron, chromSel, chrStart_locus, chrEnd_locus, trainedModelFile); 
   
   print("lastMCC ("..signedValueFunction(lastMCC)..") >= SUFFICIENT_MCC ("..signedValueFunction(SUFFICIENT_MCC)..") then break");
   stopConditionFlag = true;

  elseif lastMCC==SCORE_UNDEF and lastAccuracy >= SUFFICIENT_ACCURACY then 
   
   saveModelToFile(generalPerceptron, chromSel, chrStart_locus, chrEnd_locus, trainedModelFile); 
   
   print("lastAccuracy ("..signedValueFunction(lastAccuracy)..") >= SUFFICIENT_ACCURACY ("..signedValueFunction(SUFFICIENT_ACCURACY)..") then break");
   stopConditionFlag = true;
  end    

  return stopConditionFlag;

end

-- Function that reads a vector and replace all the occurrences of 0's to occurrences of -1's
function fromZeroOneToMinusOnePlusOne(vector)

  local newVector = {}
  for i=1,#vector do 
    newVector[i] = vector[i]
    if (vector[i] == 0) then
    newVector[i] = -1
    end
  end

  return newVector;
end


-- function that creates the ROC area under the curve
function metrics_ROC_AUC_computer(completePredValueVector, truthVector)

  -- printVector(completePredValueVector, "completePredValueVector");
  -- printVector(truthVector, "truthVector");
  -- os.exit();  
  
  if checkAllZeros(truthVector)==true then
  
  local successRate=0;	  
  io.write("ATTENTION: all the ground-truth values area 0.0 ")
  io.write("\t The metrics ROC area will be the success rate")	  
  
  local countZeros = 0
  for u=1,#completePredValueVector do
   if(completePredValueVector[u]<0.5) then countZeros = countZeros + 1; end
  end
  successRate = round(countZeros*100/#completePredValueVector, 3);	
  
  return successRate;	  
end
  

  local timeNewAreaStart0 = os.time();
  
  local tp_rate = {}
  local fp_rate = {}
  local precision_vect = {}
  local recall_vect = {}

  
  -- ROC = require '../../torch/lib/metrics/roc_thresholds.lua'
  ROC = require './lib/roc_thresholds.lua'
  local newVect = fromZeroOneToMinusOnePlusOne(truthVector)
  local roc_points = torch.Tensor(#completePredValueVector, 2)
  local precision_recall_points = torch.Tensor(#completePredValueVector, 2)
-- ######## Line too long (116 chars) ######## :
  --print("#completePredValueVector="..comma_value(#completePredValueVector).."\t#newVect="..comma_value(#newVect));
-- ######## Line too long (117 chars) ######## :
  local roc_thresholds_output = roc_thresholds(torch.DoubleTensor(completePredValueVector), torch.IntTensor(newVect))
  
  local splits = roc_thresholds_output[1]
  local thisThreshold = roc_thresholds_output[2]
  globalMinFPplusFN_vector[#globalMinFPplusFN_vector+1] = thisThreshold
  
  --print("th.\tTNs\tFNs\tTPs\tFPs\t#\tTPrate\tFPrate");
  for i = 1, #splits do
    thresholds = splits[i][1]
    tn = splits[i][2]
    fn = splits[i][3]
    tp = splits[i][4]
    fp = splits[i][5]
    
    tp_rate[i] = 0
    if ((tp+fn)~=0) then tp_rate[i] = tp / (tp+fn) end
    fp_rate[i] = 0
    if ((fp+tn)~=0) then fp_rate[i] = fp / (fp+tn) end
    
    roc_points[i][1] = tp_rate[i]
    roc_points[i][2] = fp_rate[i]
    
    precision_vect[i] = 0
    if ((tp+fp)~=0) then precision_vect[i] = tp/(tp+fp) end
    recall_vect[i]= tp_rate[i] -- = tp / (tp+fn)
    
    --io.write(round(precision_vect[i],3).." "..round(recall_vect[i],3).."\n");
    --io.flush();
    
-- ######## Line too long (103 chars) ######## :
    -- print(thresholds.."\t"..tn.."\t"..fn.."\t"..tp.."\t"..fp.."\t#\t"..tp_rate[i].."\t"..fp_rate[i])
  end

  local area_roc = round(areaNew(tp_rate,fp_rate)*100,2);
  print("metrics area_roc = "..area_roc.."%");	

	if area_roc < 0 then io.stderr:write('ERROR: AUC < 0%, problem ongoing'); return; end

	if area_roc > 100 then io.stderr:write('ERROR: AUC > 100%, problem ongoing'); return; end	
  
-- ######## Line too long (105 chars) ######## :
  -- print("#splits= "..#splits.." #precision_vect= "..#precision_vect.." #recall_vect= "..#recall_vect);
  
  
  -- printVector(precision_vect, "precision_vect");
  -- printVector(recall_vect, "recall_vect");
  
  -- require '../../torch/lib/sort_two_arrays_from_first.lua';
  require './lib/sort_two_arrays_from_first.lua';

  sortedRecallVett, sortedPrecisionVett = sort_two_arrays_from_first(recall_vect, precision_vect,  #precision_vect)
  
  -- printVector(sortedPrecisionVett, "sortedPrecisionVett");
  -- printVector(sortedRecallVett, "sortedRecallVett");
  
-- ######## Line too long (123 chars) ######## :
  local area_precision_recall = round((areaNew(sortedRecallVett, sortedPrecisionVett)-1)*100, 2) ; -- UNDERSTAND WHY -1 ???
  print("metrics AUPR area_precision_recall = "..area_precision_recall.."%");	

-- ######## Line too long (115 chars) ######## :
	if area_precision_recall < 0 then io.stderr:write('ERROR: PrecisionRecallArea < 0%, problem ongoing'); return; end
-- ######## Line too long (121 chars) ######## :
	if area_precision_recall > 100 then io.stderr:write('ERROR: PrecisionRecallArea > 100%, problem ongoing;'); return; end	
  
-- ######## Line too long (86 chars) ######## :
  printTime(timeNewAreaStart0, " the new area_roc metrics ROC_AUC_computer function");

  
  return area_roc;
end


-- Function that applies the neural network
function siameseNeuralNetwork_application(first_datasetTrain, second_datasetTrain, targetDatasetTrain, first_datasetTest, second_datasetTest, targetDatasetTest, initialPerceptron, architectureLabel, modelFile)	
  
  local time_training = os.time();
  local generalPerceptron = siameseNeuralNetwork_training(first_datasetTrain, second_datasetTrain,  targetDatasetTrain, initialPerceptron);
  
  printTime(time_training, "Training duration ");
  
  torch.save(tostring(modelFile), generalPerceptron);
  print("Saved model file: "..tostring(modelFile));

  local lastMseError = -1;
  if MINIBATCH==false then
   local zeroChar = ""
   lastMseError = rateVector[#rateVector];
   if lastMseError < 10 then zeroChar = "0" end      
   -- legend = "hidden units = "..tostring(hiddenUnits).."; hidden layers = "..tostring(hiddenLayers)..";";
   legend = " learnRate = "..tostring(LEARNING_RATE_CONST)
   legend = legend .. "; last mse = "..zeroChar..""..tostring(round(lastMseError,2)).."%"
   print("legend: "..legend)
  end

  -- %%%%%%%%%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%%%%%%%%%%
   
  local time_testing = os.time();
   local testModelOutput = testModel(first_datasetTest, second_datasetTest, targetDatasetTest, generalPerceptron)	    
   printTime(time_testing, "Testing duration ");

   local thisAccuracy = round(testModelOutput[1],2);
   local thisMCC = signedValueFunction(round(testModelOutput[3],2));
   if tonumber(thisMCC) < -1 then thisMCC="Not"; end
   if thisAccuracy < -1 then thisAccuracy="Not"; end
   architectureLabel = architectureLabel.."_Mcc"..tostring(thisMCC).."_accuracy"..tostring(thisAccuracy)
   
   if MINIBATCH==false then
    architectureLabel = architectureLabel.."_lastMse"..tostring(round(lastMseError,2)).."perc"
   end
   
   architectureLabel= string.gsub(architectureLabel, "=", "")
   
   if SAVE_MSE_VECTOR_TO_FILE then 
    
    local foldersNames = outputFileName:split("/")
    local plotVectorFolder = ""
    for i=1,(#foldersNames-1) do plotVectorFolder=tostring(plotVectorFolder)..foldersNames[i].."/"; end
 
    local vectorName = plotVectorFolder.."mseRateVector_"..tostring(os.time()).."time"..architectureLabel
    
    print("vectorName = "..vectorName)
    
    -- local vectorName = "../results/meanSquareErrors/mseRateVector_"..tostring(os.time()).."time"..architectureLabel
    
    printVectorToFile(rateVector, vectorName)
    
    local command = "Rscript ../../visualization_data/plot_single_vector.r "..vectorName
    execute_command(command)
   end
   
   print("thisMCC = "..thisMCC)
  return testModelOutput  
end


-- Function that creates the architecture of the siamese neural network
function architecture_creator(input_number, hiddenUnits, hiddenLayers, output_layer_number, dropOutFlag)

  print("Creatin\' the siamese neural network...");
  print('hiddenUnits = '..hiddenUnits..'\t hiddenLayers = '..hiddenLayers);
  
  -- imagine we have one network we are interested in, it is called "perceptronUpper"
  local perceptronUpper = nn.Sequential()
  perceptronUpper:add(nn.Linear(input_number, hiddenUnits))
  perceptronUpper:add(nn.ReLU())
  print("activation function: ReLU()")
  print("tostring(dropOutFlag) = ".. tostring(dropOutFlag))
  perceptronUpper:add(nn.Dropout()) 

  for w=1, hiddenLayers do
  perceptronUpper:add(nn.Linear(hiddenUnits,hiddenUnits))
  perceptronUpper:add(nn.ReLU())
  perceptronUpper:add(nn.Dropout())
  end

  perceptronUpper:add(nn.Linear(hiddenUnits,output_layer_number))
  --perceptronUpper:add(nn.ReLU())
  
  -- XAVIER weight initialization
  if XAVIER_INITIALIZATION ==true then 
   -- perceptronUpper = require("../../torch/lib/torch-toolbox/weight-init.lua")(perceptronUpper,  'xavier') -- XAVIER
    perceptronUpper = require("./lib/weight-init.lua")(perceptronUpper,  'xavier') -- XAVIER
  end
  
  -- local perceptronLower = perceptronUpper:clone('weight', 'gradWeight') 

  local perceptronLower= nn.Sequential()
  perceptronLower:add(nn.Linear(input_number, hiddenUnits))
  perceptronLower:add(nn.ReLU())
  perceptronLower:add(nn.Dropout())

  for w=1, hiddenLayers do
  perceptronLower:add(nn.Linear(hiddenUnits,hiddenUnits))
  perceptronLower:add(nn.ReLU())
  perceptronLower:add(nn.Dropout()) 
  end

  perceptronLower:add(nn.Linear(hiddenUnits,output_layer_number))
  -- perceptronLower:add(nn.ReLU())
  
  -- XAVIER weight initialization
  if XAVIER_INITIALIZATION ==true then 
    -- perceptronLower = require("../../torch/lib/torch-toolbox/weight-init.lua")(perceptronLower,  'xavier') -- XAVIER
    perceptronLower = require("./lib/weight-init.lua")(perceptronLower,  'xavier') -- XAVIER
  end

  
  --^^^ Highlight a cell type by setting all its starting weights to 1
  --^^^
  if HIGHLIGHT_NEURON_WEIGHT_1ST_CELL_TYPE == true and dnaseCellTypeToHighlightNumber1~=-1 and dnaseCellTypeToHighlightNumber1~="-1" then
    print("We now will highlight the cell type #"..dnaseCellTypeToHighlightNumber.." "..dnaseCellTypeToHighlightName.." by initializing to 1.0 all its weights");
    
    perceptronUpper:get(1).weight[dnaseCellTypeToHighlightNumber1]:fill(1)
    perceptronLower:get(1).weight[dnaseCellTypeToHighlightNumber1]:fill(1)
  end
  
  -- we make a parallel table that takes a pair of examples as input. they both go through the same (cloned) perceptron
  -- ParallelTable is a container module that, in its forward() method, applies the i-th member module to the i-th input, and outputs a table of the set of outputs.
  local parallel_table = nn.ParallelTable()
  parallel_table:add(perceptronUpper)
  parallel_table:add(perceptronLower)



-- ######## Line too long (112 chars) ######## :
  -- now we define our top level network that takes this parallel table and computes the cosine distance betweem
  -- the pair of outputs
  local generalPerceptron= nn.Sequential()
  generalPerceptron:add(parallel_table)
  -- generalPerceptron:add(nn.CosineDistance())
  generalPerceptron:add(NEW_CosineDistance());

  return generalPerceptron;
end



-- Training
-- ######## Line too long (118 chars) ######## :
function siameseNeuralNetwork_training(first_datasetTrain, second_datasetTrain, targetDatasetTrain, generalPerceptron)
  
  local iterations_number = ITERATIONS_CONST;
  local learnRate = LEARNING_RATE_CONST;
  
  local completionRate = 0
  local loopIterations = 1
-- ######## Line too long (88 chars) ######## :
  local trainIndexVect = {}; for i=1, #first_datasetTrain do trainIndexVect[i] = i;  end
  
  print("#trainIndexVect = "..comma_value(#trainIndexVect));

  local permutedTrainIndexVect = {};
  if PERMUTATION_TRAIN == true then
-- ######## Line too long (84 chars) ######## :
  permutedTrainIndexVect = permute(trainIndexVect, #trainIndexVect, #trainIndexVect)
  else 
  permutedTrainIndexVect = trainIndexVect
  end

  local printPercCount = 0;
  local trainDataset = {};
  local targetDataset = {};
  local errorSum = 0;
  local entroErrorSum = 0;
  
  mseSum = 0
  -- print("mseSum = "..mseSum);
  
  if MINIBATCH==true then print("MINIBATCH gradient update") end;

  print("gradient update completion rate =");
  for ite = 1, iterations_number do
  if MINIBATCH == false then
   
   -- print("training MINIBATCH == false");
   for i=1, #first_datasetTrain do  	    
    
    currentIndex = permutedTrainIndexVect[i]
    if PERMUTATION_TRAIN == true then
    io.write("(currentIndex = "..currentIndex..") ")
    io.flush();
    end
    
    completionRate = loopIterations*100/(iterations_number*#first_datasetTrain);
    if (completionRate*10)%10==0 and VERBOSE==false then
    io.write(round(completionRate,2).."% "); io.flush();
    printPercCount = printPercCount+1
    if printPercCount%10==0 then io.write("\n"); io.flush(); end
    end

-- ######## Line too long (88 chars) ######## :
   trainDataset[i]={first_datasetTrain[currentIndex], second_datasetTrain[currentIndex]}
   collectgarbage();  

   local currentTarget = 1	    
   if tonumber(targetDatasetTrain[currentIndex][1]) == 0 
   then currentTarget = -1;  end
   
   if REGULARIZATION == false then
-- ######## Line too long (113 chars) ######## :
    generalPerceptron = gradientUpdate(generalPerceptron, trainDataset[i], currentTarget, learnRate, i, ite);    
   else
-- ######## Line too long (89 chars) ######## :
    if ite==1 and i==1 then print("REGULARIZATION == true, L2_WEIGHT = "..L2_WEIGHT); end
-- ######## Line too long (127 chars) ######## :
    generalPerceptron = gradientUpdateReg(generalPerceptron, trainDataset[i], currentTarget, learnRate, L2_WEIGHT, i, ite);    
   end
   

   local predicted = generalPerceptron:forward(trainDataset[i])[1];  
   -- print("predicted = "..predicted);
   
   loopIterations = loopIterations + 1
   end
  
  else  -- MINIBATCH == true then

    -- io.write("(ite="..ite..") "); io.flush();
   
    -- print("training MINIBATCH == true");
-- ######## Line too long (128 chars) ######## :
    local output_arrangeSetsForMinibatch = arrangeSetsForMinibatch(first_datasetTrain, second_datasetTrain, targetDatasetTrain);

    local newleftTens = output_arrangeSetsForMinibatch[1];
    local newrightTens = output_arrangeSetsForMinibatch[2];
    local normTargetDatasetTrain = output_arrangeSetsForMinibatch[3];
    

    MINIBATCH_SPAN_NUMBER = math.ceil(#first_datasetTrain/MINIBATCH_SIZE);
    
    if (ite == 1) then
    print("MOMENTUM_FLAG == "..tostring(MOMENTUM_FLAG));
    print("MINIBATCH_SIZE == "..MINIBATCH_SIZE); 
    end
    
    
    local minibatch_train = {};
    local target_train = {};
    local params, gradParams = generalPerceptron:getParameters()
    local lossSum = 0
    
    local c=1
    for m=1, MINIBATCH_SPAN_NUMBER do  
     
-- ######## Line too long (83 chars) ######## :
     completionRate = loopIterations*100/(iterations_number*MINIBATCH_SPAN_NUMBER);
     if (completionRate*10)%10==0 then
      io.write(round(completionRate,2).."% "); io.flush();
      printPercCount = printPercCount+1
      if printPercCount%10==0 then io.write("\n"); io.flush(); end
     end
     
-- ######## Line too long (122 chars) ######## :
     local output_createMinibatch = createMinibatch(MINIBATCH_SIZE, m, newleftTens, newrightTens, normTargetDatasetTrain);
     local current_minibatch_train = output_createMinibatch[1];
     local current_target_train = output_createMinibatch[2];
     
     if NEW_OPTIM_GRADIENT_MINIBATCH == true then
      
      -- local function we give to optim
      -- it takes current weights as input, and outputs the loss
      -- and the gradient of the loss with respect to the weights
      -- gradParams is calculated implicitly by calling 'backward',
      -- because the model's weight and bias gradient tensors
      -- are simply views onto gradParams
       local function feval(params)
       gradParams:zero()
       local criterion = nn.MSECriterion()
     
       -- print("newMinibatch_vector["..k.."]")
       local thisPrediction = generalPerceptron:forward(current_minibatch_train)

-- ######## Line too long (89 chars) ######## :
       local loss = criterion:forward(thisPrediction, torch.Tensor{current_target_train})

       lossSum = lossSum + loss
       local error_progress = lossSum*100 / (loopIterations*MAX_POSSIBLE_MSE)
       
       if (ite%10)==0 and (m%5)==0 then
-- ######## Line too long (94 chars) ######## :
        io.write("(iteration="..ite..")(minibatch="..m..") loss = "..round(loss,2).." ")      
        io.write("\terror progress = "..round(error_progress,5).."%\n")
       end
       
-- ######## Line too long (99 chars) ######## :
       local dloss_doutput = criterion:backward(thisPrediction, torch.Tensor{current_target_train})
       generalPerceptron:backward(current_minibatch_train, dloss_doutput)

      return loss,gradParams
      end
      
      local config = {}
-- 						
      if MOMENTUM_FLAG==true and REGULARIZATION==true then
      config = {
       learningRate = LEARN_RATE,
       momentum = MOMENTUM_ALPHA,
       weightDecay = L1_WEIGHT
      }			  
      
      elseif MOMENTUM_FLAG==true and REGULARIZATION==false then
      config = {
       learningRate = LEARN_RATE,
       momentum = MOMENTUM_ALPHA
      }			  
      
      elseif MOMENTUM_FLAG==false and REGULARIZATION==true then
      config = {
       learningRate = LEARN_RATE,
       weightDecay = L1_WEIGHT
      }		
      
      else 
      config = {learningRate=LEARN_RATE}
      
      end
      -- optim.sgd(feval, params, config, state) 
      local state = nil
      
      -- if hicCellTypeTrainingSet1~="-1" and hicCellTypeTrainingSet1~=-1 then
      if HIGHLIGHT_NEURON_WEIGHT_1ST_CELL_TYPE==true then
      
	if PRINT_NOT_REG_INDICES_ONCE==true then

	print("We will now avoid the regularization on the #"..trainingSetCellTypeColumnNumber1.." cell type, corresponding to the "..hicCellTypeTrainingSet1.." cell type\n");
	io.flush()
	end
      
	optim.sgd_1cellTypeHighlight(feval, params, config, state, trainingSetCellTypeColumnNumber1, hiddenUnits, CELL_TYPE_NUMBER)
      else
	optim.sgd(feval, params, config, state) 
      end
      
      -- optim.sgd_3cellTypesHighlight(feval, params, config, state, dnaseCellTypeToHighlightNumber, hiddenUnits, CELL_TYPE_NUMBER)
      
     else  -- former gradient minibatch update
      
      local output_gradientUpdateMinibatch = gradientUpdateMinibatch(generalPerceptron, current_minibatch_train, current_target_train, learnRate, ite, c);
      
      generalPerceptron = output_gradientUpdateMinibatch[1];
      local currentError = output_gradientUpdateMinibatch[2];
      local entroError = output_gradientUpdateMinibatch[3];
      errorSum = errorSum + currentError;
      entroErrorSum = entroErrorSum + entroError
      
      if (completionRate*10)%10==0 then
-- ######## Line too long (111 chars) ######## :
      io.write("Cumulative relative MSE sum = "..round(errorSum*100/(loopIterations*MAX_POSSIBLE_MSE),2).."%");
-- ######## Line too long (115 chars) ######## :
      io.write(" cross-entropy error sum = "..round(entroErrorSum*100/(loopIterations*MAX_POSSIBLE_MSE),2).."%\n");
      io.flush();
      end
     end
     
     c = c + 1
     loopIterations = loopIterations + 1
    end
  end
  end
  

 return generalPerceptron;
end


-- function that tests the model
-- ######## Line too long (95 chars) ######## :
function testModel(first_datasetTest, second_datasetTest, targetDatasetTest, generalPerceptron)

-- ######## Line too long (84 chars) ######## :
 local testIndexVect = {}; for i=1, #first_datasetTest do testIndexVect[i] = i;  end
 local output_confusion_matrix = {};
 
 if PERMUTATION_TEST == true then
-- ######## Line too long (81 chars) ######## :
  permutedTestIndexVect = permute(testIndexVect, #testIndexVect, #testIndexVect);
 else
  permutedTestIndexVect = testIndexVect;
 end
  
 threshold = 0.5
 print("\n\n\nthreshold = "..threshold.."\n")      
 local predictionTestVect = {}
 local truthVect = {}      

 -- MINIBATCH = false
 
 --MINIBATCH=false
 print("MINIBATCH testing = "..tostring(MINIBATCH));
 
 if MINIBATCH==false then 
 
  -- print("testing MINIBATCH == false");
  for i=1, #first_datasetTest do    
  
  thisIndex = permutedTestIndexVect[i]
-- ######## Line too long (97 chars) ######## :
  --if PERMUTATION_TEST == true then io.write("(thisIndex = "..thisIndex..") "); io.flush();  end
  
-- ######## Line too long (81 chars) ######## :
  local testDataset={first_datasetTest[thisIndex], second_datasetTest[thisIndex]}
  
  -- print("#testDataset =".. #testDataset);
  	  
  local testPredictionValue = generalPerceptron:forward(testDataset)[1];
-- ######## Line too long (104 chars) ######## :
  -- print("generalPerceptron:forward(testDataset)[1] =\t "..generalPerceptron:forward(testDataset)[1]);
-- ######## Line too long (104 chars) ######## :
  -- print("generalPerceptron:forward(testDataset)[2] =\t "..generalPerceptron:forward(testDataset)[2]);

  testPredictionValue = (testPredictionValue+1)/2;
  -- print("prediction = "..round(testPredictionValue,2))
  local target = targetDatasetTest[thisIndex][1];

  predictionTestVect[#predictionTestVect+1] = testPredictionValue;
  truthVect[#truthVect+1] = target;
  end
  
 else -- MINIBATCH == true
 
  -- print("testing MINIBATCH == true");
-- ######## Line too long (123 chars) ######## :
  local output_arrangeSetsForMinibatch = arrangeSetsForMinibatch(first_datasetTest, second_datasetTest, targetDatasetTest);

  local newleftTens = output_arrangeSetsForMinibatch[1];
  local newrightTens = output_arrangeSetsForMinibatch[2];
  local normTargetDatasetTest = output_arrangeSetsForMinibatch[3];

  -- local minibatchSize = math.ceil(#first_datasetTest/MINIBATCH_SPAN_NUMBER);
  MINIBATCH_SPAN_NUMBER = math.ceil(#first_datasetTest/MINIBATCH_SIZE);
  
  print("MINIBATCH_SIZE = "..MINIBATCH_SIZE);

  local t = 1;
  print("MINIBATCH_SPAN_NUMBER = "..MINIBATCH_SPAN_NUMBER);
  
  current_model = generalPerceptron:clone()
  
  -- HACK FOR THE MINIBATCH
  current_model.modules[2] = nil --remove the cosine distance

  for m=1, MINIBATCH_SPAN_NUMBER do  
    
-- ######## Line too long (118 chars) ######## :
  local output_createMinibatch = createMinibatch(MINIBATCH_SIZE, m, newleftTens, newrightTens, normTargetDatasetTest);
  local current_minibatch_test = output_createMinibatch[1];
  local current_target_test = output_createMinibatch[2];
  	    
-- ######## Line too long (84 chars) ######## :
  -- local thisPredictionVector = generalPerceptron:forward(current_minibatch_test);
  
  -- HACK FOR THE MINIBATCH
  local doubleTensorPreCosine = current_model:forward(current_minibatch_test);
-- ######## Line too long (81 chars) ######## :
  local thisPredictionVector = siameseDistanceApplication(doubleTensorPreCosine);

  
  
  for k=1,(#thisPredictionVector)[1] do	
   predictionTestVect[#predictionTestVect+1] = (thisPredictionVector[k]+1)/2;
   truthVect[#truthVect+1] = (current_target_test[k]+1)/2;
   -- t = t + 1;
   
-- ######## Line too long (82 chars) ######## :
   -- print("(m="..m..") (k="..k..") #predictionTestVect = "..#predictionTestVect)
  end
  end
 
 end

 local printValues = false;
 
 local timeConfMat = os.time();
-- ######## Line too long (112 chars) ######## :
 output_confusion_matrix = generate_confusion_matrix(predictionTestVect, truthVect, threshold, printValues);    
 printTime(timeConfMat, "Confusion matrix computation ");
 
 metrics_ROC_AUC_computer(predictionTestVect, truthVect)
 
 lastAccuracy = output_confusion_matrix[1];
 globalArrayFPindices = output_confusion_matrix[2];
 globalArrayFPvalues = output_confusion_matrix[3];
 local lastMCC = output_confusion_matrix[4];
-- ######## Line too long (90 chars) ######## :
 --  if lastMCC ~=SCORE_UNDEF then print("lastMCC = ".. signedValueFunction(lastMCC)); end
-- ######## Line too long (84 chars) ######## :
 --  if lastAccuracy ~= SCORE_UNDEF then print("lastAccuracy = "..lastAccuracy); end
 
 return {lastAccuracy, predictionTestVect, lastMCC, truthVect};

end


-- SAVE MODEL TO FILE --
-- ######## Line too long (104 chars) ######## :
function saveModelToFile(generalPerceptronParameter, chromSel, chrStart_locus, chrEnd_locus, model_file)
 local time_id = tostring(os.time());
 
 if model_file == nil then    
-- ######## Line too long (127 chars) ######## :
  model_file = "./models/"..time_id.."_model_"..tostring(chromSel).."-"..tostring(chrStart_locus).."-"..tostring(chrEnd_locus);
  print("model_file: "..model_file);      
 end
 
 torch.save(tostring(model_file), generalPerceptronParameter);
 collectgarbage();  
 print("Saved model_file ".. model_file);
end


-- function that computes the confusion matrix
-- ######## Line too long (89 chars) ######## :
function generate_confusion_matrix(predictionTestVect, truthVect, threshold, printValues)

local tp = 0
local tn = 0
local fp = 0
local fn = 0
local MatthewsCC = SCORE_UNDEF
local accuracy = SCORE_UNDEF
local arrayFPindices = {}
local arrayFPvalues = {}
 
for i=1,#predictionTestVect do

 if printValues == true then
  io.write("predictionTestVect["..i.."] = ".. round(predictionTestVect[i],4))
  io.write("\ttruthVect["..i.."] = "..truthVect[i].." ")
  io.flush();
 end

 if predictionTestVect[i] >= threshold and truthVect[i] >= threshold then
  tp = tp + 1
  if printValues == true then print(" TP ") end
 elseif  predictionTestVect[i] < threshold and truthVect[i] >= threshold then
  fn = fn + 1
  if printValues == true then print(" FN ") end
 elseif  predictionTestVect[i] >= threshold and truthVect[i] < threshold then
  fp = fp + 1
  if printValues == true then print(" FP ") end
  arrayFPindices[#arrayFPindices+1] = i
  arrayFPvalues[#arrayFPvalues+1] = predictionTestVect[i]
 elseif  predictionTestVect[i] < threshold and truthVect[i] < threshold then
  tn = tn + 1
  if printValues == true then print(" TN ") end
 end

end

print("TOTAL:")
-- ######## Line too long (117 chars) ######## :
 print(" FN = "..comma_value(fn).." / "..comma_value(tonumber(fn+tp)).."\t (truth == 1) & (prediction < threshold)");
-- ######## Line too long (120 chars) ######## :
 print(" TP = "..comma_value(tp).." / "..comma_value(tonumber(fn+tp)).."\t (truth == 1) & (prediction >= threshold)\n");
  

-- ######## Line too long (118 chars) ######## :
 print(" FP = "..comma_value(fp).." / "..comma_value(tonumber(fp+tn)).."\t (truth == 0) & (prediction >= threshold)");
-- ######## Line too long (119 chars) ######## :
 print(" TN = "..comma_value(tn).." / "..comma_value(tonumber(fp+tn)).."\t (truth == 0) & (prediction < threshold)\n");

 local continueLabel = true

-- ######## Line too long (103 chars) ######## :
 if checkAllOnes(predictionTestVect)==true and checkAllOnes(truthVect)==false then continueLabel=false;
 print("Attention: all the predicted values are equal to 1\n");
 end
-- ######## Line too long (105 chars) ######## :
 if checkAllZeros(predictionTestVect)==true and checkAllZeros(truthVect)==false then continueLabel=false;
 print("Attention: all the predicted values are equal to 0\n");
 end
 
 if continueLabel then
  local upperMCC = (tp*tn) - (fp*fn)
  local innerSquare = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
  local lowerMCC = math.sqrt(innerSquare)
  
  if lowerMCC>0 then MatthewsCC = upperMCC/lowerMCC
  else
  MatthewsCC = SCORE_UNDEF
  end
  local signedMCC = signedValueFunction(MatthewsCC);
  
  io.write(originalChromSel.." ");
  io.write("signedMCC = "..signedMCC.." ");
  io.flush();
  if (tn > fp and tp > fn) then io.write(" excellent (tn>fp & tp>fn) "); end
  io.flush();
  
  if MatthewsCC ~= SCORE_UNDEF then 
  print("\n::::\tMatthews correlation coefficient = "..signedMCC.."\t::::\n");
  else 
  print("Matthews correlation coefficient = NOT computable");	
  end
  
  accuracy = (tp + tn)/(tp + tn +fn + fp)
-- ######## Line too long (109 chars) ######## :
  print(" accuracy = "..round(accuracy,2).. " = (tp + tn)/(tp + tn +fn + fp) \t  \t [worst = 0, best =  1]");
  
  local f1_score = SCORE_UNDEF
  if (tp+fp+fn)>0 then   
  f1_score = (2*tp) / (2*tp+fp+fn)
-- ######## Line too long (95 chars) ######## :
  print("f1_score = "..round(f1_score,2).." = (2*tp) / (2*tp+fp+fn) \t [worst = 0, best = 1]");
  else
  print("f1_score CANNOT be computed because (tp+fp+fn)==0")    
  end

  local false_discovery_rate = SCORE_UNDEF
  if (fp+tp)>0 then 
  false_discovery_rate = fp / (fp + tp)
-- ######## Line too long (111 chars) ######## :
  print("false_discovery_rate = "..round(false_discovery_rate,2).." = fp / (fp + tp) \t [worst = 1, best = 0]")
  end
  
  local precision = SCORE_UNDEF
  if (tp+fp)>0 then
  precision = tp/(tp+fp)
-- ######## Line too long (89 chars) ######## :
  print("precision = "..round(precision,2).." = tp / (tp + fp) \t [worst = 0, best = 1]")
  end
  
  local recall = SCORE_UNDEF
  if (tp+fn)>0 then
  recall = tp/(tp+fn)
-- ######## Line too long (83 chars) ######## :
  print("recall = "..round(recall,2).." = tp / (tp + fn) \t [worst = 0, best = 1]")
  end
  
  local numberOfPredictedOnes = tp + fp;
-- ######## Line too long (158 chars) ######## :
 -- print("\n\nnumberOfPredictedOnes = (TP + FP) = "..comma_value(numberOfPredictedOnes).." = "..round(numberOfPredictedOnes*100/(tp + tn + fn + fp),2).."%");
  
  io.write("\nDiagnosis: ");
  if (fn >= tp and (fn+tp)>0) then print("too many FN false negatives"); end
  if (fp >= tn and (fp+tn)>0) then print("too many FP false positives"); end
  
  
  if (tn > (10*fp) and tp > (10*fn)) then print("Excellent ! ! !");
  elseif (tn > (5*fp) and tp > (5*fn)) then print("Very good ! !"); 
  elseif (tn > (2*fp) and tp > (2*fn)) then print("Good !"); 
  elseif (tn > fp and tp > fn) then print("Alright"); 
  elseif (tn > fp and tp < fn) then print("Okay");
  elseif (tn < fp and tp > fn) then print("Okay"); 
  elseif checkAllZeros(truthVect)==false then print("Baaaad"); end
 end
 
 return {accuracy, arrayFPindices, arrayFPvalues, MatthewsCC};
end




-- -- training
-- ######## Line too long (105 chars) ######## :
-- function gradientUpdate_minibatchBIS(perceptron, dataset, target, learningRate, ite, minibatch_number)
-- 
--   print("gradientUpdate_minibatchBIS()")
--   
--   realTarget=changeSignOfArray(target)
--   
--   target_array_tensors = torch.Tensor(realTarget)
-- 
--   predictionValue = perceptron:forward(dataset);
--   local mseVect = {}
--   local mseSum = 0
--   
-- --   current_prediction = predictionValue
-- --   current_model = perceptron
-- --   current_dataset = dataset
-- --   if 0==0 then return nil; end
--   
--   for p=1,(#predictionValue)[1] do
--    print('predictionValue['..p..'] = '..predictionValue[p]);
--     mseVect[p] = math.pow(target[p] - predictionValue[p],2);
--     mseSum = mseSum + mseVect[p];
--   end  
--   local averageMse = round(mseSum/((#predictionValue)[1]),2);
--  
-- ######## Line too long (113 chars) ######## :
--   print("(ite = "..ite..") (minibatch_number = "..minibatch_number..") minibatch average mse = "..averageMse);
--     
--     gradientWrtOutput = target_array_tensors
--     perceptron:zeroGradParameters() 
--     perceptron:backward(dataset, gradientWrtOutput) 
--     perceptron:updateParameters(learningRate)
-- 
--     
--     
--   return perceptron;
--   
-- end


-- Gradient update for the siamese neural network with minibatch
-- ######## Line too long (118 chars) ######## :
function gradientUpdateMinibatch(generalPerceptron, dataset_vector, targetVector, learningRate, ite, minibatch_number)

 function dataset_vector:size() return #dataset_vector end
 local target_array_tensors = changeSignOfArray(targetVector)  
 local gradientWrtOutput = torch.Tensor(target_array_tensors)   
 
 local normPredictionValue = (generalPerceptron:forward(dataset_vector)+1)/2
 local normTargetVector = {}
 for u=1,#targetVector do normTargetVector[u]=(targetVector[u]+1)/2; end
 
 local mseVect = {}
 local mseSum = 0
 
 local entroErrorVect = {}
 local entroErrorSum = 0
 
-- ######## Line too long (84 chars) ######## :
--     print("((dataset_vector[1]):size())[2] = "..((dataset_vector[1]):size())[2]);
--     print("(#normPredictionValue)[1] = ".. (#normPredictionValue)[1])
-- ######## Line too long (85 chars) ######## :
--     print("normPredictionValue:size()[1] = "..normPredictionValue:size()[1].."\n")
 
 for p=1,normPredictionValue:size()[1] do
 -- print('normPredictionValue['..p..'] = '..normPredictionValue[p]);
  mseVect[p] = math.pow(normTargetVector[p] - normPredictionValue[p],2);
  mseSum = mseSum + mseVect[p];
  
-- ######## Line too long (162 chars) ######## :
  io.write("(p="..p..") normPredictionValue["..p.."] = "..round(normPredictionValue[p],2).."\t normTargetVector["..p.."] = "..round(normTargetVector[p],2).."\t");
-- ######## Line too long (90 chars) ######## :
  entroErrorVect[p] = -(math.log(normPredictionValue[p])*normTargetVector[p]); -- TO CHECK
 --  print("entroErrorVect["..p.."] = "..round(entroErrorVect[p]));
  entroErrorSum = entroErrorSum + entroErrorVect[p]; -- TO CHECK
 end  
 -- local averageMse = mseSum  -- / ((#normPredictionValue)[1]);

-- ######## Line too long (100 chars) ######## :
 -- print("(ite = "..ite..") (minibatch = "..minibatch_number..") average mse = "..round(mseSum,3));
 

 generalPerceptron:zeroGradParameters();
 generalPerceptron:backward(dataset_vector, gradientWrtOutput);
 generalPerceptron:updateParameters(learningRate);
 
--     current_model = generalPerceptron
--     current_dataset = dataset_vector
--     current_prediction = normPredictionValue

 return {generalPerceptron, mseSum, entroErrorSum};
end

VERBOSE = false

-- Gradient update for the siamese neural network with regularization
-- ######## Line too long (106 chars) ######## :
function gradientUpdateReg(generalPerceptron, input_profile, targetValue, learningRate, l2_weight, i, ite)

 function input_profile:size() return #input_profile end
 local predictionValue = generalPerceptron:forward(input_profile)[1]

 local regPenalty = 0  -- Regularization error

 if predictionValue*targetValue < 1 then
  gradientWrtOutput = torch.Tensor({-targetValue})
  generalPerceptron:zeroGradParameters();
  generalPerceptron:backward(input_profile, gradientWrtOutput)
  generalPerceptron:updateParameters(learningRate)
  local parameters, _ = generalPerceptron:parameters()
 
  for i=1, table.getn(parameters) do
  parameters[i]:mul(1-l2_weight)  -- updating parameters with L2 regularization
-- ######## Line too long (94 chars) ######## :
  regPenalty = regPenalty + l2_weight * parameters[i]:norm(2) -- updating regularization error
  end
 end

 local meanSquareError = math.pow(targetValue - predictionValue,2)
-- ######## Line too long (101 chars) ######## :
 local totalError = meanSquareError + regPenalty -- total_error = data_error + regularization_penalty

 
 if i%50==0 and ite%20==0 and VERBOSE==true  then
-- ######## Line too long (138 chars) ######## :
  io.write("(ite="..ite..") (ele="..i..") pred = "..signedValueFunction(predictionValue).." target = "..signedValueFunction(targetValue));
-- ######## Line too long (114 chars) ######## :
  io.write(" => totalError = "..round(totalError,3).." = "..round(meanSquareError,3).." + "..round(regPenalty,3));
  
  io.write("\n");
  io.flush();
 end
 
 count = ite*DATA_SIZE+i;
 mseSum = mseSum + totalError;
   
 -- it's 50 because the error goes from 0 to 4
 mseSumRate = mseSum*100/(count*MAX_POSSIBLE_MSE)
 rateVector[#rateVector+1] = mseSumRate; 
 rateIndexVector[#rateIndexVector+1] = count;  
  	    
 if i%50==0 and ite%20==0 and VERBOSE==true then
-- ######## Line too long (126 chars) ######## :
  io.write("(ite="..ite..") (ele="..i..") mseSumRate = (mseSum + totalError)*100/"..count.." = "..round(mseSumRate,2).."%\n");
 end
 
 return generalPerceptron  
end


-- Gradient update for the siamese neural network
-- ######## Line too long (92 chars) ######## :
function gradientUpdate(generalPerceptron, input_profile, targetValue, learningRate, i, ite)

 function input_profile:size() return #input_profile end   
 local predictionValue = generalPerceptron:forward(input_profile)[1];
 
 local meanSquareError = math.pow(targetValue - predictionValue,2);
 
 if (ite%PRINT_NUMBER==0 or i%PRINT_NUMBER==0) and VERBOSE==true  then
-- ######## Line too long (163 chars) ######## :
  io.write("(ite="..ite..") (ele="..i..") pred = "..signedValue(predictionValue).." target = "..signedValue(targetValue) .." => mse = "..round(meanSquareError,3));
  io.flush();
 end
 
 count = ite*DATA_SIZE+i;
 mseSum = mseSum + meanSquareError;
  
 if ite == iterations_number then
-- ######## Line too long (98 chars) ######## :
  print("mseSum = mseSum + meanSquareError = "..round(mseSum,2).." + "..round(meanSquareError,2));
 end
  
 -- it's 50 because the error goes from 0 to 4
 mseSumRate = mseSum*100/(count*MAX_POSSIBLE_MSE)
 rateVector[#rateVector+1] = mseSumRate; 
 rateIndexVector[#rateIndexVector+1] = count;
  
-- ######## Line too long (144 chars) ######## :
 -- print("mseSum = "..comma_value(round(mseSum,2)).." / "..comma_value(count*MAX_POSSIBLE_MSE).. "\t mseSumRate = "..round(mseSumRate,2).."%");

 if predictionValue*targetValue < 1 then
  gradientWrtOutput = torch.Tensor({-targetValue});
  generalPerceptron:zeroGradParameters();
  generalPerceptron:backward(input_profile, gradientWrtOutput);          
  
  if MOMENTUM_FLAG==false then  
   generalPerceptron:updateParameters(learningRate);
  else
  if i==1 and ite==1 then print("MOMENTUM_FLAG == true") end	
-- ######## Line too long (83 chars) ######## :
  local currentParameters, currentGradParameters = generalPerceptron:parameters();	
-- ######## Line too long (98 chars) ######## :
  generalPerceptron:momentumUpdateParameters(learningRate, MOMENTUM_ALPHA, currentGradParameters);
  end      
 end

return generalPerceptron;
end




-- TO LAUNCH:
-- ######## Line too long (220 chars) ######## :
-- number=$(( ( RANDOM % 100000 )  + 1 )); label="withoutClone_Relu_shuffleTest_2"; qsub -q hoffmangroup -N siamese_nn_toy -cwd -b y -o ./output\_$number\_$label -e ./output\_$number\_$label th siamese_nn_toy.lua $label;


print('\n\n @ @ @ @ @ @ START @ @ @ @ @ @ @ ');
print('file: siamese_nn_toy.lua');
print('author Davide Chicco <davide.chicco@gmail.com>');
print(os.date("%c", os.time()));

timeStart = os.time()


require "nn";

-- # # # DATA READING # # #


-- require "../../_project/bin/database_management.lua"
-- require "../../_project/bin/utils.lua"
require "./lib/database_management.lua"
require "./lib/utils.lua"


io.write(">>> th siamese_nn_toy.lua ");
MAX_PARAMS = #arg
for i=1,MAX_PARAMS do io.write(" "..tostring(arg[i]).." "); end
io.write("\n\n\n");
io.flush();


local label = tostring(arg[1]);
print("Test label = "..label);
local tuple_limit = tonumber(arg[2]);
print("tuple_limit = "..tuple_limit);
local locus_position_limit = 500000
local balancedFlag = false
local chromSel = tostring(arg[3]);
originalChromSel = chromSel
local chrStart_locus = tonumber(arg[4])
local chrEnd_locus = tonumber(arg[5]) 
local dataSource = "Thurman_Miriam"
local original_tuple_limit = 100
local balancedFalsePerc = tonumber(arg[6])
print("balancedFalsePerc = "..balancedFalsePerc.."%");
CELL_TYPE_NUMBER = 82

TRAINING_SAMPLES_PERC = tonumber(arg[7])
local percSymbol = "%"
if TRAINING_SAMPLES_PERC == -1 then percSymbol="" end 
io.write("TRAINING_SAMPLES_PERC = "..TRAINING_SAMPLES_PERC..percSymbol.." ");

-- ######## Line too long (86 chars) ######## :
print("training segment: "..chromSel.." from "..chrStart_locus.." to "..chrEnd_locus);


outputFileName = tostring(arg[8]);
local execution = tostring(arg[9]); -- "OPTIMIZATION-TRAINING-HELD-OUT"
local trainedModelFile = tostring(arg[10]); 
local retrieveFP_flag_ini = tostring(arg[11]); -- true or false
local retrieveFP_flag = false;

if retrieveFP_flag_ini == "TRUE" or retrieveFP_flag_ini == "true" 
then retrieveFP_flag = true;
elseif retrieveFP_flag_ini == "FALSE" or retrieveFP_flag_ini == "false" 
then retrieveFP_flag = false;
else 
 print("retrieveFP_flag_ini must be TRUE or FALSE; the program will stop");
 os.exit();
end

print("retrieveFP_flag = "..tostring(retrieveFP_flag));

local val_chrStart_locus = -1
local val_chrEnd_locus = -1
local val_tuple_limit = -1
local val_balancedFalsePerc = -1

val_chrStart_locus = tonumber(arg[12]) 
val_chrEnd_locus = tonumber(arg[13]) 
val_tuple_limit = tonumber(arg[14]) 
val_balancedFalsePerc = tonumber(arg[15]) 
  
-- ######## Line too long (96 chars) ######## :
print("validation segment: "..chromSel.." from "..val_chrStart_locus.." to "..val_chrEnd_locus);
print("val_tuple_limit = "..val_tuple_limit);
print("val_balancedFalsePerc = "..val_balancedFalsePerc.."%");


local secondSpan_chrStart_locus = -1 
local secondSpan_chrEnd_locus = -1


-- ######## Line too long (268 chars) ######## :
if execution == "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT" or execution == "OPTIMIZATION-TRAINING-CROSS-VALIDATION-DOUBLE-INPUT" or execution == "SINGLE-MODEL-TRAINING-HELD-OUT-DISTAL" or execution == "SINGLE-MODEL-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT" then

  secondSpan_chrStart_locus = tonumber(arg[16]) 
  secondSpan_chrEnd_locus = tonumber(arg[17]) 
  print("secondSpan_chrStart_locus = "..secondSpan_chrStart_locus);
  print("secondSpan_chrEnd_locus = "..secondSpan_chrEnd_locus);
end

MINIBATCH = tostring(arg[18]) 
if MINIBATCH == "true" or MINIBATCH == "TRUE" 
then MINIBATCH = true;
else MINIBATCH = false;
end

print("MINIBATCH = "..tostring(MINIBATCH));

-- MINIBATCH_SIZE = 20 -- working
MINIBATCH_SIZE = tonumber(arg[19])
print("MINIBATCH_SIZE = ".. MINIBATCH_SIZE)

print("ITERATIONS_CONST = "..ITERATIONS_CONST); 
print("LEARNING_RATE_CONST = "..LEARNING_RATE_CONST);


dnaseExcludeColumn = tonumber(arg[20])
print("dnaseExcludeColumn = "..dnaseExcludeColumn)

dnaseExcludeColumnName = "";
local columnNames = {}
if dnaseExcludeColumn>=1 and dnaseExcludeColumn<=CELL_TYPE_NUMBER then
columnNames = getColumnNamesOfTable("chromregionprofiles")
dnaseExcludeColumnName = columnNames[dnaseExcludeColumn]

print("EXCLUDING THE FEATURE-COLUMN "..dnaseExcludeColumnName.." number "..dnaseExcludeColumn.." among "..CELL_TYPE_NUMBER);
elseif dnaseExcludeColumn==-1 or dnaseExcludeColumn=="-1" then
print("No cell type will be excluded from the input DNase table");
else

 print("Error: the dnaseExcludeColumnName = "..dnaseExcludeColumn.." is not in the 1- "..CELL_TYPE_NUMBER.." interval. The program will stop");
os.exit();  
end

local hicCellTypeValidSet1 = tostring(arg[21]);
print("hicCellTypeValidSet1 = "..hicCellTypeValidSet1);
local hicCellTypeValidSet2 = tostring(arg[22]);
print("hicCellTypeValidSet2 = "..hicCellTypeValidSet2);
local hicCellTypeValidSet3 = tostring(arg[23]);
print("hicCellTypeValidSet3 = "..hicCellTypeValidSet3);
local hicCellTypeValidSet4 = tostring(arg[24]);
print("hicCellTypeValidSet4 = "..hicCellTypeValidSet4);


dnaseCellTypeToHighlightName1 = "";
dnaseCellTypeToHighlightNumber1, dnaseCellTypeToHighlightName1 = retrieveCellTypeColumnNameAndNumber(hicCellTypeValidSet1)
dnaseCellTypeToHighlightName2 = "";
dnaseCellTypeToHighlightNumber2, dnaseCellTypeToHighlightName2 = retrieveCellTypeColumnNameAndNumber(hicCellTypeValidSet2)
dnaseCellTypeToHighlightName3 = "";
dnaseCellTypeToHighlightNumber3, dnaseCellTypeToHighlightName3 = retrieveCellTypeColumnNameAndNumber(hicCellTypeValidSet3)
dnaseCellTypeToHighlightName4 = "";
dnaseCellTypeToHighlightNumber4, dnaseCellTypeToHighlightName4 = retrieveCellTypeColumnNameAndNumber(hicCellTypeValidSet4)


if tonumber(dnaseCellTypeToHighlightNumber1)~=-1 then
  NO_INTERSECTION_BETWEEN_SETS = false
end

PROFI_FLAG = false
PROFI_FLAG = tostring(arg[25])
if tostring(PROFI_FLAG) == tostring(true) then
ProFi = require "../temp/ProFi"
print("ProFi = require ../temp/ProFi")
ProFi:start()
print("ProFi:start()")
end


hicCellTypeTrainingSet1 = tostring(arg[26]); -- 	THIS IS ALSO A FLAG TO UNDERSTAND IF THE TRAINING SET IS CELL-TYPE-SPECIFIC
print("hicCellTypeTrainingSet1 = ".. hicCellTypeTrainingSet1)
hicCellTypeTrainingSet2 = tostring(arg[27]); -- 	THIS IS ALSO A FLAG TO UNDERSTAND IF THE TRAINING SET IS CELL-TYPE-SPECIFIC
print("hicCellTypeTrainingSet2 = ".. hicCellTypeTrainingSet2)
hicCellTypeTrainingSet3 = tostring(arg[28]); -- 	THIS IS ALSO A FLAG TO UNDERSTAND IF THE TRAINING SET IS CELL-TYPE-SPECIFIC
print("hicCellTypeTrainingSet3 = ".. hicCellTypeTrainingSet3)
hicCellTypeTrainingSet4 = tostring(arg[29]); -- 	THIS IS ALSO A FLAG TO UNDERSTAND IF THE TRAINING SET IS CELL-TYPE-SPECIFIC
print("hicCellTypeTrainingSet4 = ".. hicCellTypeTrainingSet4)

trainingSetCellTypeColumnName1 = "";
trainingSetCellTypeColumnNumber1 = -1;
if hicCellTypeTrainingSet1~="-1" and hicCellTypeTrainingSet1~=-1 then
  print("[input] The training set will be made of Hi-C interactions of "..hicCellTypeTrainingSet1.." cell type");
  trainingSetCellTypeColumnNumber1, trainingSetCellTypeColumnName1 = retrieveCellTypeColumnNameAndNumber(hicCellTypeTrainingSet1)
end

trainingSetCellTypeColumnName2 = "";
trainingSetCellTypeColumnNumber2 = -1;
if hicCellTypeTrainingSet2~="-1" and hicCellTypeTrainingSet2~=-1 then
  print("[input] The training set will be made of Hi-C interactions of "..hicCellTypeTrainingSet2.." cell type");
  trainingSetCellTypeColumnNumber2, trainingSetCellTypeColumnName2 = retrieveCellTypeColumnNameAndNumber(hicCellTypeTrainingSet2)
end


trainingSetCellTypeColumnName3 = "";
trainingSetCellTypeColumnNumber3 = -1;
if hicCellTypeTrainingSet3~="-1" and hicCellTypeTrainingSet3~=-1 then
  print("[input] The training set will be made of Hi-C interactions of "..hicCellTypeTrainingSet3.." cell type");
  trainingSetCellTypeColumnNumber3, trainingSetCellTypeColumnName3 = retrieveCellTypeColumnNameAndNumber(hicCellTypeTrainingSet3)
end


trainingSetCellTypeColumnName4 = "";
trainingSetCellTypeColumnNumber4 = -1;
if hicCellTypeTrainingSet4~="-1" and hicCellTypeTrainingSet4~=-1 then
  print("[input] The training set will be made of Hi-C interactions of "..hicCellTypeTrainingSet4.." cell type");
  trainingSetCellTypeColumnNumber4, trainingSetCellTypeColumnName4 = retrieveCellTypeColumnNameAndNumber(hicCellTypeTrainingSet4)
end



-- % -- % -- % -- % -- % -- % -- End of the input reading -- % -- % -- % -- % -- % 

if execution ~=  "OPTIMIZATION-TRAINING-HELD-OUT" 
and execution ~= "OPTIMIZATION-TRAINING-CROSS-VALIDATION" 
and execution ~= "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL"
and execution ~= "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT"
and execution ~= "OPTIMIZATION-TRAINING-CROSS-VALIDATION-DOUBLE-INPUT"
and execution ~= "SINGLE-MODEL-TRAINING-HELD-OUT-DISTAL"
and execution ~= "SINGLE-MODEL-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT"
and execution ~= "SINGLE-MODEL-TRAINING-CROSS-VALIDATION" 
and execution ~= "JUST-TESTING"
and execution ~= "BOOSTING-TESTING" then

print("Error: execution is wrong! The program will stop!");
os.exit();
end


local regionLabel = chromSel.."-"..chrStart_locus.."-"..chrEnd_locus;

INDEPENDENT_VALIDATION_DATASET_READING = true


dataset_firstChromRegion = {}
dataset_secondChromRegion = {}
targetVector = {}

val_dataset_firstChromRegion = {}
val_dataset_secondChromRegion = {}
val_targetVector = {}
local dnaseDataTable_only_IDs_training = {}
local dnaseDataTable_only_IDs_val = {}

local dnaseDataTable = {}
local val_dnaseDataTable = {}

if READ_DATA_FROM_DB == true then

if execution ~= "JUST-TESTING" then

experimentDetails = "==> Experiment details:\n "..regionLabel.."\n";
experimentDetails = experimentDetails .." tuple_limit = "..tuple_limit.."\n";
experimentDetails = experimentDetails .." percentage of (-1) negative elements in the training set and test set = "..balancedFalsePerc.."%\n";
experimentDetails = experimentDetails .." percentage of (+1) positive elements in the training set and test set = "..tonumber(100-balancedFalsePerc).."%\n";
experimentDetails = experimentDetails .." percentage of elements for the training set = "..TRAINING_SAMPLES_PERC.."%\n";
experimentDetails = experimentDetails .." percentage of elements for the test set = "..tonumber(100-TRAINING_SAMPLES_PERC).."%\n";

local uniformDistribution = true;

-- READIN' THE TRAINING SET

-- previous (one cell type)
-- local unbal_data_read_output = readDataThroughPostgreSQL_segment(chromSel, tuple_limit, locus_position_limit, balancedFlag, chrStart_locus, chrEnd_locus, execution, CELL_TYPE_NUMBER, dataSource, balancedFalsePerc, uniformDistribution, dnaseExcludeColumn, hicCellTypeValidSet, hicCellTypeTrainingSet)

local unbal_data_read_output = readDataThroughPostgreSQL_segment(chromSel, tuple_limit, locus_position_limit, balancedFlag, chrStart_locus, chrEnd_locus, execution, CELL_TYPE_NUMBER, dataSource, balancedFalsePerc, uniformDistribution, dnaseExcludeColumn, hicCellTypeValidSet1, hicCellTypeValidSet2, hicCellTypeValidSet3, hicCellTypeValidSet4, hicCellTypeTrainingSet1, hicCellTypeTrainingSet2, hicCellTypeTrainingSet3, hicCellTypeTrainingSet4)



local balancedDatasetSize = unbal_data_read_output[1];    
-- print("balancedDatasetSize ".. comma_value(balancedDatasetSize));
dnaseDataTable = unbal_data_read_output[2]; -- dnaseDataTable is the unbalanced test set

dnaseDataTable_only_IDs_training = unbal_data_read_output[8];
  
 if balancedDatasetSize==0 then
-- ######## Line too long (82 chars) ######## :
  print("No true interactions in the training set: the program is going to stop");
  os.exit();
 end
  
dataset_firstChromRegion = unbal_data_read_output[3];
dataset_secondChromRegion = unbal_data_read_output[4];
targetVector = unbal_data_read_output[5];

end --(endif execution ~= "JUST-TESTING")



-- READIN' THE VALIDATION SET
if execution == "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL" or execution == "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT" or execution == "SINGLE-MODEL-TRAINING-HELD-OUT-DISTAL" or execution == "SINGLE-MODEL-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT" or execution == "SINGLE-MODEL-TRAINING-CROSS-VALIDATION" or execution == "JUST-TESTING"  then
 
 if INDEPENDENT_VALIDATION_DATASET_READING == true then
  
  local val_uniformDistribution = true
  
  
  if hicCellTypeValidSet1~="-1" and hicCellTypeValidSet1~=-1 then
    print("The validation set will contain interactions of the "..hicCellTypeValidSet1.." cell type");
  end
  if hicCellTypeValidSet2~="-1" and hicCellTypeValidSet2~=-1 then
    print("The validation set will contain interactions of the "..hicCellTypeValidSet2.." cell type");
  end
  if hicCellTypeValidSet3~="-1" and hicCellTypeValidSet3~=-1 then
    print("The validation set will contain interactions of the "..hicCellTypeValidSet3.." cell type");
  end
  if hicCellTypeValidSet4~="-1" and hicCellTypeValidSet4~=-1 then
    print("The validation set will contain interactions of the "..hicCellTypeValidSet4.." cell type");
  end
  
  -- In the training set, we read interactions of the hicCellTypeTrainingSet cell type and avoided hicCellTypeValidSet cell types
  -- Now in the test set, we do the opposite
 
  -- previous (one cell type):
  -- local val_dataset_output = readDataThroughPostgreSQL_segment(chromSel, val_tuple_limit, locus_position_limit, balancedFlag, val_chrStart_locus, val_chrEnd_locus, execution, CELL_TYPE_NUMBER, dataSource, val_balancedFalsePerc, val_uniformDistribution, dnaseExcludeColumn, hicCellTypeTrainingSet, hicCellTypeValidSet)
  
  local val_dataset_output = readDataThroughPostgreSQL_segment(chromSel, val_tuple_limit, locus_position_limit, balancedFlag, val_chrStart_locus, val_chrEnd_locus, execution, CELL_TYPE_NUMBER, dataSource, val_balancedFalsePerc, val_uniformDistribution, dnaseExcludeColumn, hicCellTypeTrainingSet1, hicCellTypeTrainingSet2, hicCellTypeTrainingSet3, hicCellTypeTrainingSet4, hicCellTypeValidSet1, hicCellTypeValidSet2, hicCellTypeValidSet3, hicCellTypeValidSet4)
  
  val_dnaseDataTable = val_dataset_output[2]; 
  val_dataset_firstChromRegion = val_dataset_output[3];
  val_dataset_secondChromRegion = val_dataset_output[4];
  val_targetVector = val_dataset_output[5];  
  
  dnaseDataTable_only_IDs_val = val_dataset_output[8];
   

 else
  VAL_PERC = 20
  
-- ######## Line too long (140 chars) ######## :
  print("since INDEPENDENT_VALIDATION_DATASET_READING is set to false, the validation set will be the "..VAL_PERC.."% of the training set");
  
  
  local val_tuple_limit = round((tuple_limit*VAL_PERC)/100,0);
  
  
  -- We arrange the validation set
  val_dnaseDataTable = subtable(dnaseDataTable, (#dnaseDataTable-val_tuple_limit+1), #dnaseDataTable)
  
  val_dataset_firstChromRegion = subtable(dataset_firstChromRegion, (#dataset_firstChromRegion-val_tuple_limit+1), #dataset_firstChromRegion)
  
  val_dataset_secondChromRegion = subtable(dataset_secondChromRegion, (#dataset_secondChromRegion-val_tuple_limit+1), #dataset_secondChromRegion)
  
  val_targetVector = subtable(targetVector, (#targetVector-val_tuple_limit+1), #targetVector)
  
  dnaseDataTable_only_IDs_val = subtable(dnaseDataTable_only_IDs_training, (#dnaseDataTable_only_IDs_training-val_tuple_limit+1), #dnaseDataTable_only_IDs_training)
  
  -- We rearrange the training set
  temp_dnaseDataTable = subtable(dnaseDataTable, 1, (#dnaseDataTable-val_tuple_limit))

  temp_dataset_firstChromRegion = subtable(dataset_firstChromRegion, 1, (#dataset_firstChromRegion-val_tuple_limit))

  temp_dataset_secondChromRegion = subtable(dataset_secondChromRegion, 1, (#dataset_secondChromRegion-val_tuple_limit))

  temp_targetVector = subtable(targetVector, 1, (#targetVector-val_tuple_limit))

  dnaseDataTable_only_IDs_temp = subtable(dnaseDataTable_only_IDs_training, 1, (#dnaseDataTable_only_IDs_training-val_tuple_limit))
  
  -- We reassign the training set
  dnaseDataTable = temp_dnaseDataTable
  dataset_firstChromRegion = temp_dataset_firstChromRegion
  dataset_secondChromRegion = temp_dataset_secondChromRegion
  targetVector = temp_targetVector
  dnaseDataTable_only_IDs_training = dnaseDataTable_only_IDs_temp
  
  
 end
end

 -- READIN' THE 2nd input training SET
if execution == "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT" or execution == "OPTIMIZATION-TRAINING-CROSS-VALIDATION-DOUBLE-INPUT" then
 
    local secondSpan_dataset_output = readDataThroughPostgreSQL_segment(chromSel, tuple_limit, locus_position_limit, balancedFlag, secondSpan_chrStart_locus, secondSpan_chrEnd_locus, execution, CELL_TYPE_NUMBER, dataSource, balancedFalsePerc, uniformDistribution, dnaseExcludeColumn, hicCellTypeValidSet)
    
    secondSpan_dataset_firstChromRegion = secondSpan_dataset_output[3];
    secondSpan_dataset_secondChromRegion = secondSpan_dataset_output[4];
    secondSpan_targetVector = secondSpan_dataset_output[5];      
    
    dataset_firstChromRegion =  tableConcat(dataset_firstChromRegion, secondSpan_dataset_firstChromRegion);
    dataset_secondChromRegion =  tableConcat(dataset_secondChromRegion, secondSpan_dataset_secondChromRegion);
    targetVector =  tableConcat(targetVector, secondSpan_targetVector);
  
end

else

    require "../data/2016-04-05_datafiles/chr21-46562780-47409790matrix_file_1459864690time_RIGHT.lua"
    require "../data/2016-04-05_datafiles/chr21-46562780-47409790matrix_file_1459864690time_LEFT.lua"
    require "../data/2016-04-05_datafiles/chr21-46562780-47409790matrix_file_1459864690time_LABELS.lua"

    dataset_firstChromRegion = first_datasetGeneral
    dataset_secondChromRegion = second_datasetGeneral
    targetVector = targetDatasetGeneral

    require "../data/2016-04-05_datafiles/chr21-15630020-16234310matrix_file_1459874054time_LEFT.lua"
    require "../data/2016-04-05_datafiles/chr21-15630020-16234310matrix_file_1459874054time_RIGHT.lua"
    require "../data/2016-04-05_datafiles/chr21-15630020-16234310matrix_file_1459874054time_LABELS.lua"

    val_dataset_firstChromRegion = first_datasetGeneral
    val_dataset_secondChromRegion = second_datasetGeneral
    val_targetVector = targetDatasetGeneral

    -- TO REMOVE
    -- dataset_firstChromRegion = val_dataset_firstChromRegion
    -- dataset_secondChromRegion = val_dataset_secondChromRegion
    -- targetVector = val_targetVector 



    print("> > > > Data were read from files,  not from PostgreSQL < < < <");


end

if dnaseExcludeColumn >= 1 and dnaseExcludeColumn <= CELL_TYPE_NUMBER then
  CELL_TYPE_NUMBER = CELL_TYPE_NUMBER -1
  print("siamese_nn_toy.lua: global CELL_TYPE_NUMBER = "..CELL_TYPE_NUMBER)
end

local noIntersectiontimeStart = os.time()



if (hicCellTypeValidSet1~=-1 and hicCellTypeValidSet1~="-1") or execution=="JUST-TESTING" then
  NO_INTERSECTION_BETWEEN_SETS = false
end

--	REMOVING DATA FROM THE VALIDATION SET
if NO_INTERSECTION_BETWEEN_SETS==true then

 -- Remove the test set elements from the training set
 print("Remove the test set elements from the training set");
 print("before removal:")
 print("BEFORE #val_dnaseDataTable = "..comma_value(#val_dnaseDataTable))
-- ######## Line too long (93 chars) ######## :
 print("BEFORE #val_dataset_firstChromRegion = "..comma_value(#val_dataset_firstChromRegion))
-- ######## Line too long (95 chars) ######## :
 print("BEFORE #val_dataset_secondChromRegion = "..comma_value(#val_dataset_secondChromRegion))
 print("BEFORE #val_targetVector = "..comma_value(#val_targetVector))

 local index = 1
 local _size = #dnaseDataTable  
 local kRate = 1
 for k=1,_size do
  
  kRate=(k*100/_size)
  if (kRate*10)%10==0 then 
   io.write(kRate.."% "); io.flush(); 
  end
  
-- ######## Line too long (89 chars) ######## :
  local output_contains = chromRegionTableContains(val_dnaseDataTable, dnaseDataTable[k])
  local label = output_contains[1]
  --print("label = "..tostring(label))
  local position = output_contains[2]
  -- print("position = "..position)
  if label  then
  io.write("(k="..comma_value(k).." remove)\t");
  io.flush();
  table.remove(val_dnaseDataTable, position);
  table.remove(val_dataset_firstChromRegion, position);
  table.remove(val_dataset_secondChromRegion, position);
  table.remove(val_targetVector, position);
  index = 1


  else
  --io.write("\n");
  index = index +1 
  end  	
 end
 
 
-- ######## Line too long (116 chars) ######## :
printTime(noIntersectiontimeStart, " removal of the elements which were present both in training set and test set");

print("after removal:")
print("AFTER #val_dnaseDataTable = "..comma_value(#val_dnaseDataTable))
-- ######## Line too long (91 chars) ######## :
print("AFTER #val_dataset_firstChromRegion = "..comma_value(#val_dataset_firstChromRegion))
-- ######## Line too long (93 chars) ######## :
print("AFTER #val_dataset_secondChromRegion = "..comma_value(#val_dataset_secondChromRegion))
print("AFTER #val_targetVector = "..comma_value(#val_targetVector))
  
end


print("\nAFTER #dnaseDataTable = "..comma_value(#dnaseDataTable))
-- ######## Line too long (83 chars) ######## :
print("AFTER #dataset_firstChromRegion = "..comma_value(#dataset_firstChromRegion))
-- ######## Line too long (85 chars) ######## :
print("AFTER #dataset_secondChromRegion = "..comma_value(#dataset_secondChromRegion))
print("AFTER #targetVector = "..comma_value(#targetVector))



DATA_SIZE = #dataset_firstChromRegion;
print("DATA_SIZE = ".. comma_value(DATA_SIZE));

print("Data reading phase finished, let's close the connection");
closeGlobalDbConnection();


print("== Data reading recap ==")
print("The training set contains Hi-C interactions of")
if (hicCellTypeTrainingSet1~="-1" and hicCellTypeTrainingSet1~=-1) then print(hicCellTypeTrainingSet1) end
if (hicCellTypeTrainingSet2~="-1" and hicCellTypeTrainingSet2~=-1) then print(hicCellTypeTrainingSet2) end
if (hicCellTypeTrainingSet3~="-1" and hicCellTypeTrainingSet3~=-1) then print(hicCellTypeTrainingSet3) end
if (hicCellTypeTrainingSet4~="-1" and hicCellTypeTrainingSet4~=-1) then print(hicCellTypeTrainingSet4) end
print("The test set contains Hi-C interactions of")
if (hicCellTypeValidSet1~="-1" and hicCellTypeValidSet1~=-1) then print(hicCellTypeValidSet1) end
if (hicCellTypeValidSet2~="-1" and hicCellTypeValidSet2~=-1) then print(hicCellTypeValidSet2) end
if (hicCellTypeValidSet3~="-1" and hicCellTypeValidSet3~=-1) then print(hicCellTypeValidSet2) end
if (hicCellTypeValidSet4~="-1" and hicCellTypeValidSet4~=-1) then print(hicCellTypeValidSet4) end
print("== == == == == ==")

-- print("The program will stop")
-- os.exit()


PRINT_NUMBER = 1
if DATA_SIZE >= 1000 then PRINT_NUMBER = round(DATA_SIZE/500);
else PRINT_NUMBER = round(DATA_SIZE/10); end
-- print("PRINT_NUMBER = "..PRINT_NUMBER);

print("MINIBATCH = "..tostring(MINIBATCH));
print("XAVIER_INITIALIZATION = "..tostring(XAVIER_INITIALIZATION));
print("MOMENTUM_FLAG = "..tostring(MOMENTUM_FLAG));


rateVector = {};
rateIndexVector = {};
mseSum = 0;

local first_datasetTrain = {}
local second_datasetTrain = {}
local targetDatasetTrain = {}

local first_datasetTest = {}
local second_datasetTest = {}
local targetDatasetTest = {}

local vectorMCC = {}
local vectorAccuracy = {}

local stopConditionFlag = false

-- TRAINING_SAMPLES = math.floor(DATA_SIZE*TRAINING_SAMPLES_PERC/100)
-- print("TRAINING_SAMPLES = "..comma_value(TRAINING_SAMPLES));

local dropOutFlag = true

input_number = -1

-- ######## Line too long (305 chars) ######## :
if execution == "OPTIMIZATION-TRAINING-HELD-OUT" or execution == "OPTIMIZATION-TRAINING-CROSS-VALIDATION" or execution == "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL" or execution == "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT" or execution == "OPTIMIZATION-TRAINING-CROSS-VALIDATION-DOUBLE-INPUT" then


local largeMseCount = 0; 
local generalPerceptronToSave = {};

-- local hiddenUnits = 4
-- local hiddenLayers = 4

local maxHiddenLayers = 4
local maxHiddenUnits = 50

local hiddenUnitsVect = {}
hiddenUnitsVect[#hiddenUnitsVect+1] = 600 -- 300
hiddenUnitsVect[#hiddenUnitsVect+1] = 650 -- 300
hiddenUnitsVect[#hiddenUnitsVect+1] = 700 -- 200
hiddenUnitsVect[#hiddenUnitsVect+1] = 750 -- 300
hiddenUnitsVect[#hiddenUnitsVect+1] = 800 -- 300

for hiddenLayers=1,maxHiddenLayers do
 for _,v in ipairs(hiddenUnitsVect) do
  
  rateVector = {};
  rateIndexVector = {};
  mseSum = 0;      
  
  hiddenUnits = v;

  input_number = CELL_TYPE_NUMBER
  output_layer_number = CELL_TYPE_NUMBER
  
  io.write(" dropOutFlag = "..tostring(dropOutFlag));
  io.write(" hiddenUnits = "..hiddenUnits);
  io.write(" hiddenLayers = "..hiddenLayers.."\n");
  io.flush();
  
-- ######## Line too long (162 chars) ######## :
  local architectureLabel = tostring("_"..regionLabel.."_tuples="..tuple_limit.."_hiddenUnits="..tostring(hiddenUnits).."_hiddenLayers="..tostring(hiddenLayers));

-- ######## Line too long (126 chars) ######## :
  local initialPerceptron = architecture_creator(input_number, hiddenUnits, hiddenLayers, output_layer_number, dropOutFlag);  
  
-- ######## Line too long (133 chars) ######## :
  if execution == "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL" or execution == "OPTIMIZATION-TRAINING-HELD-OUT-DISTAL-DOUBLE-INPUT" then		

  local first_datasetTrain = dataset_firstChromRegion
  local second_datasetTrain = dataset_secondChromRegion
  local targetDatasetTrain = targetVector

  local first_datasetTest = val_dataset_firstChromRegion
  local second_datasetTest = val_dataset_secondChromRegion
  local targetDatasetTest = val_targetVector
  
-- ######## Line too long (236 chars) ######## :
  local applicationOutput = siameseNeuralNetwork_application(first_datasetTrain, second_datasetTrain, targetDatasetTrain, first_datasetTest, second_datasetTest, targetDatasetTest, initialPerceptron, architectureLabel, trainedModelFile);
  
  vectorAccuracy[#vectorAccuracy+1] = applicationOutput[1];
  vectorMCC[#vectorMCC+1] = applicationOutput[3];	
  
  local currentMCC = applicationOutput[3]
  local currentAccuracy = applicationOutput[1]
  io.write("currentMCC = "..signedValueFunction(round(currentMCC,3)));
  io.write(" currentAccuracy = "..round(currentAccuracy,3));
  io.write(" hiddenUnits = "..hiddenUnits);
  io.write(" hiddenLayers = "..hiddenLayers.."\n");
  io.flush();
  
  -- printVector(vectorAccuracy, "PARTIAL vectorAccuracy");
  printVector(vectorMCC, "PARTIAL vectorMCC");
  
-- ######## Line too long (88 chars) ######## :
  local stopCondition = continueOrStopCheck(applicationOutput[1], applicationOutput[3]);
  
  if stopCondition==true then break; end

 elseif execution == "OPTIMIZATION-TRAINING-HELD-OUT" then		

-- ######## Line too long (84 chars) ######## :
  local first_datasetTrain = subrange(dataset_firstChromRegion, 1, TRAINING_SAMPLES)
-- ######## Line too long (86 chars) ######## :
  local second_datasetTrain = subrange(dataset_secondChromRegion, 1, TRAINING_SAMPLES)
  local targetDatasetTrain = subrange(targetVector, 1, TRAINING_SAMPLES)

-- ######## Line too long (93 chars) ######## :
  local first_datasetTest = subrange(dataset_firstChromRegion, TRAINING_SAMPLES+1, DATA_SIZE)
-- ######## Line too long (95 chars) ######## :
  local second_datasetTest = subrange(dataset_secondChromRegion, TRAINING_SAMPLES+1, DATA_SIZE)
-- ######## Line too long (81 chars) ######## :
  local targetDatasetTest = subrange(targetVector, TRAINING_SAMPLES+1, DATA_SIZE)
  
-- ######## Line too long (235 chars) ######## :
  local applicationOutput = siameseNeuralNetwork_application(first_datasetTrain, second_datasetTrain, targetDatasetTrain, first_datasetTest, second_datasetTest, targetDatasetTest, initialPerceptron, architectureLabel, trainedModelFile)
  
  vectorAccuracy[#vectorAccuracy+1] = applicationOutput[1]
  vectorMCC[#vectorMCC+1] = applicationOutput[3]
  -- printVector(vectorAccuracy, "PARTIAL vectorAccuracy")
  printVector(vectorMCC, "PARTIAL vectorMCC")
  
  print("applicationOutput[3] = "..applicationOutput[3])
  
-- ######## Line too long (87 chars) ######## :
  local stopCondition = continueOrStopCheck(applicationOutput[1], applicationOutput[3])

  if stopCondition==true then break; end  
  
-- ######## Line too long (136 chars) ######## :
 elseif execution == "OPTIMIZATION-TRAINING-CROSS-VALIDATION" or execution == "OPTIMIZATION-TRAINING-CROSS-VALIDATION-DOUBLE-INPUT" then
  
  local globalPredictionVector = {}
  local truthVector = {}

-- ######## Line too long (161 chars) ######## :
  local kfold_output = kfold_cross_validation(dataset_firstChromRegion, dataset_secondChromRegion, targetVector, initialPerceptron, architectureLabel, DATA_SIZE)
  
  globalPredictionVector = kfold_output[1]
  truthVector = kfold_output[2]

  local threshold = 0.5		

  print("$$$ K-fold cross validation finished, general rates $$$")
  -- metrics_ROC_AUC_computer(globalPredictionVector, truthVector)
  local printValues = false
-- ######## Line too long (122 chars) ######## :
  local output_confusion_matrix = generate_confusion_matrix(globalPredictionVector, truthVector, threshold, printValues)  

  vectorAccuracy[#vectorAccuracy+1] = output_confusion_matrix[1]
  vectorMCC[#vectorMCC+1] = output_confusion_matrix[4]
  printVector(vectorAccuracy, "PARTIAL vectorAccuracy")
  printVector(vectorMCC, "PARTIAL vectorMCC")
  
-- ######## Line too long (99 chars) ######## :
  local stopCondition = continueOrStopCheck(output_confusion_matrix[1], output_confusion_matrix[4])

  if stopCondition==true then break; end  

  end         
  
 end -- END HIDDEN UNITS LOOP
 
 if stopCondition==true then print("stopCondition==true then break") break; end
 
end -- END HIDDEN LAYERS LOOP

elseif execution == "SINGLE-MODEL-TRAINING-HELD-OUT-DISTAL"  then

  hiddenUnits = 600 -- USUALLY 100
  hiddenLayers = 1 -- USUALLY 1

  local input_number = CELL_TYPE_NUMBER
  local output_layer_number = CELL_TYPE_NUMBER
  
  io.write(" dropOutFlag = "..tostring(dropOutFlag));
  io.write(" hiddenUnits = "..hiddenUnits);
  io.write(" hiddenLayers = "..hiddenLayers.."\n");
  io.flush();
  
-- ######## Line too long (162 chars) ######## :
  local architectureLabel = tostring("_"..regionLabel.."_tuples="..tuple_limit.."_hiddenUnits="..tostring(hiddenUnits).."_hiddenLayers="..tostring(hiddenLayers));

-- ######## Line too long (126 chars) ######## :
  local initialPerceptron = architecture_creator(input_number, hiddenUnits, hiddenLayers, output_layer_number, dropOutFlag);  
  
  local first_datasetTrain = dataset_firstChromRegion
  local second_datasetTrain = dataset_secondChromRegion
  local targetDatasetTrain = targetVector

  local first_datasetTest = val_dataset_firstChromRegion
  local second_datasetTest = val_dataset_secondChromRegion
  local targetDatasetTest = val_targetVector
  
  
  local timeSiameseNeuralNetwork_application = os.time();
  
-- ######## Line too long (236 chars) ######## :
  local applicationOutput = siameseNeuralNetwork_application(first_datasetTrain, second_datasetTrain, targetDatasetTrain, first_datasetTest, second_datasetTest, targetDatasetTest, initialPerceptron, architectureLabel, trainedModelFile);
  
  local currentMCC = applicationOutput[3]
  local currentAccuracy = applicationOutput[1]
  io.write("currentMCC = "..round(currentMCC,2));
  io.write(" currentAccuracy = "..round(currentAccuracy,3));

  io.write(" hiddenUnits = "..hiddenUnits);
  io.write(" hiddenLayers = "..hiddenLayers.."\n");
  io.flush();
  
-- ######## Line too long (98 chars) ######## :
  printTime(timeSiameseNeuralNetwork_application, "Siamese neural network application duration ");

  
elseif execution == "SINGLE-MODEL-TRAINING-CROSS-VALIDATION" then
  
  hiddenUnits = 200  --  it was 100
  hiddenLayers = 1

  local input_number = CELL_TYPE_NUMBER
  local output_layer_number = CELL_TYPE_NUMBER
  
  io.write(" dropOutFlag = "..tostring(dropOutFlag));
  io.write(" hiddenUnits = "..hiddenUnits);
  io.write(" hiddenLayers = "..hiddenLayers.."\n");
  io.flush();
  
-- ######## Line too long (162 chars) ######## :
  local architectureLabel = tostring("_"..regionLabel.."_tuples="..tuple_limit.."_hiddenUnits="..tostring(hiddenUnits).."_hiddenLayers="..tostring(hiddenLayers));

-- ######## Line too long (124 chars) ######## :
  local initialPerceptron = architecture_creator(input_number, hiddenUnits, hiddenLayers, output_layer_number, dropOutFlag);


  local globalPredictionVector = {}
  local truthVector = {}

-- ######## Line too long (162 chars) ######## :
  local kfold_output = kfold_cross_validation(dataset_firstChromRegion, dataset_secondChromRegion, targetVector, initialPerceptron, architectureLabel, DATA_SIZE);
  
  globalPredictionVector = kfold_output[1];
  truthVector = kfold_output[2];

  local threshold = 0.5		

  print("$$$ K-fold cross validation finished, general rates $$$");
  -- metrics_ROC_AUC_computer(globalPredictionVector, truthVector)
  local printValues = false;
-- ######## Line too long (122 chars) ######## :
  local output_confusion_matrix = generate_confusion_matrix(globalPredictionVector, truthVector, threshold, printValues)  
  
  
-- ######## Line too long (139 chars) ######## :
-- 	local testModelOutput = testModel(val_dataset_firstChromRegion, val_dataset_secondChromRegion, targetDatasetTest, QUALE PERCEPTRON????)

elseif execution == "JUST-TESTING" then

print("#val_dnaseDataTable = ".. #val_dnaseDataTable)
print("#val_dataset_firstChromRegion =".. #val_dataset_firstChromRegion)
print("#val_dataset_secondChromRegion ="..#val_dataset_secondChromRegion)

print("modelFile to read: ".. (trainedModelFile));
io.flush();
local loadedPerceptron =  torch.load(trainedModelFile);

print("\n\n\n : : : : : EXTERNAL TEST  : : : : : ");

--   local first_datasetTest = val_dataset_firstChromRegion
--   local second_datasetTest = val_dataset_secondChromRegion
--   local targetDatasetTest = val_targetVector

local timeTestModel = os.time();
-- ######## Line too long (130 chars) ######## :
local testModelOutput = testModel(val_dataset_firstChromRegion, val_dataset_secondChromRegion, val_targetVector, loadedPerceptron)
printTime(timeTestModel, "Testing duration ");

elseif execution == "BOOSTING-TESTING" then

first_datasetTest = dataset_firstChromRegion
second_datasetTest = dataset_secondChromRegion
targetDatasetTest = targetVector

local testModelOutput = {}
local predictionTestVectList = {}
local threshold = 0.5
local truthVect = {} 
local boostingTrainedModelFile = {}

-- ######## Line too long (89 chars) ######## :
boostingTrainedModelFile[1] = "./models/chr21-15630020-28144290_IMBAL-NEG_trained_model";
-- ######## Line too long (84 chars) ######## :
boostingTrainedModelFile[2] = "./models/chr21-36793160-39866830_BALA_trained_model";
-- ######## Line too long (89 chars) ######## :
boostingTrainedModelFile[3] = "./models/chr21-35021440-36789490_IMBAL-POS_trained_model";

MAX_MODELS = 3
for i=1,MAX_MODELS do

 print("\n : : : : : BOOSTING EXTERNAL TEST  : : : : : ");
 print("modelFile to read: ".. (boostingTrainedModelFile[i]));
 local loadedPerceptron =  torch.load(boostingTrainedModelFile[i]);       

 local timeTestModel = os.time();
-- ######## Line too long (108 chars) ######## :
 testModelOutput[i] = testModel(first_datasetTest, second_datasetTest, targetDatasetTest, loadedPerceptron);
 
 predictionTestVectList[i] = testModelOutput[i][2];
 truthVect = testModelOutput[i][4];
 
-- ######## Line too long (81 chars) ######## :
 -- testModel(): return {lastAccuracy, predictionTestVect, lastMCC, truthVect};  
 printTime(timeTestModel, "Testing duration ");   
end

local majorityPredictionVect = {}

print("\n : : : : : BOOSTING MAJORITY TEST  : : : : : ");
-- compute the majority
for j=1,#(predictionTestVectList[1]) do    
-- ######## Line too long (164 chars) ######## :
  majorityPredictionVect[j] = majorityGreaterThanThreshold(predictionTestVectList[1][j], predictionTestVectList[2][j], predictionTestVectList[3][j], threshold);    
end

local printValues = false
-- ######## Line too long (118 chars) ######## :
local output_confusion_matrix = generate_confusion_matrix(majorityPredictionVect, truthVect, threshold, printValues)  

end




printVector(vectorAccuracy, "FINAL vectorAccuracy");
printVector(vectorMCC, "FINAL vectorMCC");

-- RETRIEVE THE FALSE POSITIVES -- 
if #globalArrayFPindices > 0 and retrieveFP_flag==true then

 print("#globalArrayFPindices = "..#globalArrayFPindices)
 print("#dataset_firstChromRegion = "..#dataset_firstChromRegion)
 print("#dataset_secondChromRegion = "..#dataset_secondChromRegion)
 print("chromSel = "..chromSel)
 print("#dnaseDataTable = "..#dnaseDataTable)
 print("first_profile_initial = "..first_profile_initial)
 print("first_profile_finish = "..first_profile_finish)
 print("second_profile_initial = "..second_profile_initial)
 print("second_profile_finish = "..second_profile_finish)
 print("last_index = "..last_index)

 
-- ######## Line too long (279 chars) ######## :
local final_interaction_list = retrieveInteractionsDetails_Thruman2012data(globalArrayFPindices, dataset_firstChromRegion, dataset_secondChromRegion, chromSel, dnaseDataTable, first_profile_initial, first_profile_finish, second_profile_initial, second_profile_finish, last_index)

-- ######## Line too long (145 chars) ######## :
if final_interaction_list==nil then print("main() final_interaction_list production equals to nil, Error. The program will stop"); os.exit(); end
  
-- ######## Line too long (88 chars) ######## :
fromInteractionIdsToChromRegions(chromSel, final_interaction_list, globalArrayFPvalues);

else print("No false positives (FP) in this round"); 

end

-- RENAME THE FILE --

-- ######## Line too long (104 chars) ######## :
local renameFileCommand = "mv ./"..tostring(outputFileName).." "..tostring(outputFileName).."_FINISHED";
execute_command(renameFileCommand)


if tostring(PROFI_FLAG) == tostring(true) then 
ProFi:stop()
print("ProFi:stop()")
ProFi:writeReport('../temp/myProfilingReport'..tostring(os.time()))
print("ProFi:writeReport()")
end

printTime(timeStart, "Total duration ");

print(os.date("%c", os.time()));
print('file: siamese_nn_toy.lua');
print('\n\n @ @ @ @ @ @ END @ @ @ @ @ @ @ ');



  -- %%%%%%%%%%%%%%%%%%%%% VISUALIZATION %%%%%%%%%%%%%%%%%%%%%%%

  -- plotYmax = 100
  -- plotYmin = 0
  -- 
  -- require 'gnuplot';
  -- plotFileName = "mseRate_"..os.time().."time.pdf"
  -- gnuplot.pdffigure(plotFileName)
  -- gnuplot.axis{1,#rateVector,plotYmin,plotYmax}
  -- rateTensor = torch.Tensor{rateIndexVector, rateVector};
  -- gnuplot.grid(true);
  --  gnuplot.plot({'Cumulative mean square error',rateTensor[2]})
-- ######## Line too long (141 chars) ######## :
  --  gnuplot.xlabel('iterations  (= '..tostring(comma_value(ITERATIONS_CONST))..') and elements (= '..tostring(comma_value(DATA_SIZE))..')')
  -- gnuplot.ylabel('Mean square error rate %')
  -- 
  -- 
  -- DATA_SIZE = #first_datasetTrain	--  gnuplot.title(legend)
  -- -- gnuplot.plotflush()



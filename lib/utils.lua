
-- folder=../results/2015-10-25_PostgreSQL_parallelization/
-- model_file=./models/PROVA_MODEL4
-- th utils.lua chr21 30 $folder -1 -1 -1 "Thurman2012" -1 -1 $model_file

--MORDOR_QUEUE = "all.q"
MORDOR_QUEUE = "hoffmangroup"


-- Function to concat a tensor and an array by position (a1, b1, a2, b2, a3, b3, ...)
function tensorArrayConcatByPos(te,a)
  local t_merge = {}
    for i=1,(te:size())[1] do
        t_merge[#t_merge+1] = te[i]
	t_merge[#t_merge+1] = a[i]
	i = i + 1
    end
    return torch.Tensor(t_merge)
end

-- Function to concat two arrays by position (a1, b1, a2, b2, a3, b3, ...)
function arrayConcatByPos(t1,t2)
  local t_merge = {}
    for i=1,#t1 do
        t_merge[#t_merge+1] = t1[i]
	t_merge[#t_merge+1] = t2[i]
	i = i + 1
    end
    return t_merge
end

-- Function tableCopy
function tableCopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[tableCopy(orig_key)] = tableCopy(orig_value)
        end
        setmetatable(copy, tableCopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end


-- Function 
function toNumberIfPossible(value)
  local numValue = tonumber(value)
  if numValue==nil then return value 
  else return numValue end;
end

-- Print the weights of the neural network
function printWeights(nn_weigthts)
   local original_t = nn_weigthts
   local nn_weigthts = nn_weigthts:view(-1)
   for i=1,nn_weigthts:nElement() do      
     io.write(round(nn_weigthts[i],4));
      if (i%original_t:size()[2]==0) then io.write("\n"); 
      else io.write(','); end
   end
end

-- Print the weights of the neural network
function printWeightsToFile(nn_weigthts, fileName)
  
   local outputFile = assert(io.open(fileName, "w"))
   local original_t = nn_weigthts
   local nn_weigthts = nn_weigthts:view(-1)
   for i=1,nn_weigthts:nElement() do      
      --io.write(round(nn_weigthts[i],4));
      outputFile:write(round(nn_weigthts[i],4));
      if (i%original_t:size()[2]==0) 
	then 
	  -- io.write("\n"); 
	  outputFile:write("\n"); 
	else 
	  -- io.write(','); 
	  outputFile:write(','); 
      end
   end
   outputFile:close();
end


-- FUNCTION subrange
function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

-- Permutations
-- tab = {1,2,3,4,5,6,7,8,9,10}
-- permute(tab, 10, 10)
function permute(tab, n, count)
      n = n or #tab
      for i = 1, count or n do
        local j = math.random(i, n)
        tab[i], tab[j] = tab[j], tab[i]
      end
      return tab
end

-- from sam_lie
-- Compatible with Lua 5.0 and 5.1.
-- Disclaimer : use at own risk especially for hedge fund reports :-)

---============================================================
-- add comma to separate thousands
-- 
function comma_value(amount)
  local formatted = amount
  while true do  
    formatted, k = string.gsub(formatted, "^(-?%d+)(%d%d%d)", '%1,%2')
    if (k==0) then
      break
    end
  end
  return formatted
end

-- round a real value
function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end




-- Fucntion toboolean()
function toboolean(X)
   return not not X
end

-- Function table.contains
function table.contains(table, element)  
  local count = 1
  for _, value in pairs(table) do
   -- print("value: "..tostring(value).." element: "..tostring(element));
    if tostring(value) == tostring(element) or value==element then
      return {true,count}
    end
    count = count + 1
  end
  return {false,-1}
end

-- Function chrom region table contains
function chromRegionTableContains(table, element)  
  
  local count = 1
  for i=1,#table do
    -- let's check chrom region number, chrom start, chrom end
    if tonumber(table[i][1])==tonumber(element[1]) and tonumber(table[i][2])==tonumber(element[2]) and tonumber(table[i][3])==tonumber(element[3]) then return  {true,count} end
     count = count + 1
  end  
  
  return {false,-1}
end


-- Function tableConcat(t1,t2)
function tableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

-- change the sign of an array
function changeSignOfArray(array)
  
   local newArray = {}
    for i=1,#array do
      newArray[i] = - (array[i])
           
     -- newArray[i] =  (array[i])            
      -- io.write("\t"..array[i].." BECOMES "..newArray[i].."\n");
      -- io.flush();
    end

    return newArray;
end

-- Command that checks the equivalence between two tensors
-- 1) with eq(), it puts to 1 all the elements that are the same in the two tensors
-- 2) then it counts the number of 1's with the sum function
-- 3) finally, it checks if the 1's count is the same of the number of elements in the input tensors
function haveTwoTensorsTheSameContent(tens1, tens2)
       -- return tens1:eq(tens2):sum() == tens1:size()[1]
      return (torch.all(torch.eq(tens1, tens2)));
end

-- function that prints a signed value
function signedValue(value)
  if value>0 then io.write("+"..value) 
  else io.write(value) end
end
       

-- function that checks if all the elements of a vector are +1
function checkAllOnes(vector)
  
  local dime = #vector
  local total = 0
  local flag = true
  
  for i=1,dime do
    if round(vector[i],0)~=1 then flag = false end
    total = total + vector[i]
  end
  
  --print("vector average="..round((total/dime),2));
  
  return flag;
end

-- function that checks if all the elements of a vector are 0
function checkAllZeros(vector)
  
  local dime = #vector
  local total = 0
  local flag = true
  
  for i=1,dime do
    if round(vector[i],0)~=0 then flag = false end
    total = total + vector[i]
  end
  
  --print("vector average="..round((total/dime),2));
  
  return flag;
end

-- Permutations
-- tab = {1,2,3,4,5,6,7,8,9,10}
-- permute(tab, 10, 10)
function permute(tab, n, count)
      n = n or #tab
      for i = 1, count or n do
        local j = math.random(i, n)
        tab[i], tab[j] = tab[j], tab[i]
      end
      return tab
end

-- Function string:split()
function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

-- Function that reads a value and returns the string of the signed value
function signedValueFunction(value)
  
      local value = tonumber(value);
      --print("value = "..value);
      local charPlus = ""
      if tonumber(value) >= 0 then charPlus = "+"; end
      local outputString = charPlus..""..tostring(round(value,2));
      --print("outputString = "..outputString);
      
      return tostring(outputString);
end


  
-- Function that prints a vector to standard output
function printVector(vector, title)
  print("printin' the vector named: ".. title);
  for i=1,#vector do
        
     if (string.match(title, "MCC") or string.match(title, "mcc")) then
       
       if vector[i]==-2 then
	  io.write("vector["..i.."] = NOT COMPUTABLE \n"); 
       else
	  io.write("vector["..i.."] = "..signedValueFunction(vector[i]).." \n"); 
       end
     else
	  io.write("vector["..i.."] = "..round(tonumber(vector[i]),2).." \n");   
     end
       io.flush();
  end
end

  
-- Function that prints a vector to a file
function printVectorToFile(vector, fileName)
  
  local file = assert(io.open(fileName, "a"));
  
  print("printin' the vector named: "..fileName);
  for i=1,#vector do
     file:write(tonumber(vector[i])); 
     
     if i<#vector then file:write("\n"); end
     file:flush();
  end
  
  file:close();
end


  
  

	

-- check if all the elements of a vector are true
function checkAllTrues(vettNew)
  local result = true;
  for _,v in ipairs(vettNew) do 
      if v~=true then return false; end 
  end
  return result;
end

-- print a value with name
function printValue(amount, word)
  print(word.." "..comma_value(amount));  
end


-- add comma to separate thousands
function comma_value(amount)
  local formatted = amount
  while true do  
    formatted, k = string.gsub(formatted, "^(-?%d+)(%d%d%d)", '%1,%2')
    if (k==0) then
      break
    end
  end
  return formatted
end

-- subtable function
function subtable(table, lower_index, upper_index)

  return_table = {}
  k = 1
  for i=lower_index,upper_index do
      return_table[k] = table[i]
      k = k+1
  end

return return_table;

end


-- Command creator
function command_creator(index_start, index_end, folder, chromSel, dropOutFlag, hiddenUnits, hiddenLayers, inputThreshold, dataSource, toy_dataset, limit_number, parallelization, model_file, main_execution_file, execution_mode, tagLabel, predictionVector_file, recordedMatrixFile, retrieveFPs, val_index_start, val_index_end, momentumParameter, validation_limit_number, training_balancedFalsePerc, validation_balancedFalsePerc)
  
  torch.manualSeed(torch.random(1,10000))
  
  local label = tostring(os.time()).."time_"..tostring(torch.random(1,10000).."_"..tostring(tagLabel));

  local output_file_name = folder..chromSel.."_output_siamese_parallel_"..label;
      
  local command = "qsub -q "..tostring(MORDOR_QUEUE).." -N "..chromSel.."_siamese_segment_"..label.." -cwd -b y -o " ..output_file_name.." -e " ..output_file_name.." th ".. main_execution_file.." "..chromSel.." "..dropOutFlag.." "..execution_mode.." "..hiddenUnits.." "..hiddenLayers.." "..inputThreshold.." "..dataSource.."  "..toy_dataset.." "..limit_number.." ".. index_start .." ".. index_end .." "..parallelization.." "..model_file;
  
  command = command .." ".. predictionVector_file.." "..recordedMatrixFile.." "..retrieveFPs.." "..val_index_start.." "..val_index_end.." "..momentumParameter .." "..validation_limit_number.." " ..training_balancedFalsePerc.." "..validation_balancedFalsePerc;
  
  return command;
end

-- Execute the command
function execute_command(command)

  local handle = io.popen(command);
  local result = handle:read("*a");
  print("command result: \t "..result);
  handle:close();

end

-- Delete file if it exists
function delete_file(fileName)
  
  -- DELETE MODEL FILE IF PRESENT
  if(file_exists(fileName)) then
      io.write(fileName.." file already present: the system will delete it\n");
      io.flush();  
      
      command = "rm "..fileName;
      print("command: \t "..command);

      
      handle = io.popen(command);
      result = handle:read("*a");
      print("command result: \t "..result);
      handle:close(); 
  end  
end

-- Function that loops until the file exists
function wait_until_the_file_exists(fileName)
  
  io.write("Does the "..fileName.. " file exist? ");
  io.flush();
  local p=1
  -- check if the file exists
  while(true) do
    if(file_exists(fileName)) then
      io.write("YES!\n");
      
      os.execute("sleep " .. tonumber(1))
      
      io.flush();  
      return true;
    end

    if(p%1000000000==0) then io.write("Not yet... "); end
    io.flush();
    p = p + 1  
  end
  
  
end

-- Function that prints 
function printTime(timeStart, stringToPrint)
	timeEnd = os.time();
	duration = timeEnd - timeStart;
	print('\nduration '..stringToPrint.. ': '.. comma_value(tonumber(duration)).. ' seconds');
	io.flush();
	print('duration '..stringToPrint.. ': '..string.format("%.2d days, %.2d hours, %.2d minutes, %.2d seconds", (duration/(60*60))/24, duration/(60*60)%24, duration/60%60, duration%60)) 
	io.flush();
	
    return duration;
end

-- File exists
function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

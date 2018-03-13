require 'nn'
local NEW_CosineDistance, parent = torch.class('NEW_CosineDistance', 'nn.Module')

-- Function myCosineApplication()
function myCosineApplication(tensorA, tensorB)
  
  local result = -2;

  local A_size = (#tensorA)[1]
  local B_size = (#tensorB)[1]
  
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

-- Constructor
function NEW_CosineDistance:__init()
   parent.__init(self)   
   self.gradInput = {torch.Tensor(), torch.Tensor()}
end 

local function makeContiguous(self, input1, input2)
   if not input1:isContiguous() then
      self._input1 = self._input1 or input1.new()
      self._input1:resizeAs(input1):copy(input1)
      input1 = self._input1
   end
   if not input2:isContiguous() then
      self._input2 = self._input2 or input2.new()
      self._input2:resizeAs(input2):copy(input2)
      input2 = self._input2
   end
   return input1, input2
end



function NEW_CosineDistance:updateOutput(input)
   local input1, input2 = input[1], input[2]

   input1, input2 = makeContiguous(self, input1, input2)
   
   if input1:dim() == 1 then
      input1 = input1:view(1,-1)
      input2 = input2:view(1,-1)
   end

   if not self.buffer then
      self.buffer = input1.new()
      self.w1  = input1.new()
      self.w22 = input1.new()
      self.w  = input1.new()
      self.w32 = input1.new()
      self.ones = input1.new()
   end

   self.buffer:cmul(input1,input2)
   self.w1:sum(self.buffer,2)

   local epsilon = 1e-12
   self.buffer:cmul(input1,input1)
   self.w22:sum(self.buffer,2):add(epsilon)
   self.ones:resizeAs(self.w22):fill(1)
   self.w22:cdiv(self.ones, self.w22)
   self.w:resizeAs(self.w22):copy(self.w22)

   self.buffer:cmul(input2,input2)
   self.w32:sum(self.buffer,2):add(epsilon)
   self.w32:cdiv(self.ones, self.w32)
   self.w:cmul(self.w32)
   self.w:sqrt()

   self.output:cmul(self.w1,self.w)
   self.output:resize(input1:size(1))

   -- NEW PART   
   
    local resultVector = {} 
    local lun = (#input1)[1]
    -- print("lun = "..lun)
    
    for i=1,lun do
      resultVector[#resultVector+1] = myCosineApplication(input1[i], input2[i])      
    end
      
      return torch.Tensor(resultVector);
end

function NEW_CosineDistance:updateGradInput(input, gradOutput)
   local v1  = input[1]
   local v2  = input[2]
   local not_batch = false
   
   v1, v2 = makeContiguous(self, v1, v2)
   
   if v1:dim() == 1 then
      v1 = v1:view(1,-1)
      v2 = v2:view(1,-1)
      not_batch = true
   end

   if #self.gradInput ~= 2 then
      self.gradInput[1] = self.gradInput[1] or v1.new()
      self.gradInput[2] = self.gradInput[2] or v1.new()
   end

   local gw1 = self.gradInput[1]
   local gw2 = self.gradInput[2]
   gw1:resizeAs(v1):copy(v2)
   gw2:resizeAs(v1):copy(v1)

   self.buffer:cmul(self.w1,self.w22)
   gw1:addcmul(-1,self.buffer:expandAs(v1),v1)
   gw1:cmul(self.w:expandAs(v1))

   self.buffer:cmul(self.w1,self.w32)
   gw2:addcmul(-1,self.buffer:expandAs(v1),v2)
   gw2:cmul(self.w:expandAs(v1))

   local go = gradOutput:view(-1,1):expandAs(v1)
   gw1:cmul(go)
   gw2:cmul(go)

   if not_batch then
      self.gradInput[1]:resize(gw1:size(2))
      self.gradInput[2]:resize(gw2:size(2))
   end

   return self.gradInput
end

function NEW_CosineDistance:clearState()
   nn.utils.clear(self, {
      'buffer',
      'w1',
      'w22',
      'w',
      'w32',
      'ones',
   })
   return parent.clearState(self)
end
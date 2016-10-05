
local BiasTable, parent = torch.class('nn.BiasTable', 'nn.Module')

function BiasTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function BiasTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])

   if self.output:dim() == 4 then
      if input[2]:dim() ~= 2 then
         error('if input is batched, bias must be batched')
      end

      local batchSize = self.output:size()[1]
      local planes = self.output:size()[2]

      for batch=1,batchSize do
         for plane=1,planes do
            self.output[{batch,plane,{},{}}]:add(input[2][{batch,plane}])
         end
      end
   else
      if input[2]:dim() ~= 1 then
         error('bias cannot be batched if input is not batched')
      end

      local planes = self.output:size()[1]
      for plane=1,planes do
         self.output[{plane,{},{}}]:add(input[2][plane])
      end
   end
   return self.output
end

function BiasTable:updateGradInput(input, gradOutput)
   self.gradInput[1]:copy(gradOutput)
   self.gradInput[2] = self.gradInput[2] or input[2]:new()

   if gradOutput.dim() == 4 then
      local batchSize = gradOutput:size()[1]
      local planes = gradOutput:size()[2]

      for batch=1,batchSize do
         for plane=1,planes do
            self.gradInput[2][{batch,plane}] = self.gradInput[2][{batch,plane}] + self.gradOutput[{batch,plane,{},{}}]:sum()
         end
      end
   else
      local planes = self.gradInput[1]:size()[1]
      for plane=1,planes do
         self.gradInput[2][plane] = self.gradInput[2][plane] + self.gradOutput[{plane,{},{}}]:sum()
      end
   end

   for i=3, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end

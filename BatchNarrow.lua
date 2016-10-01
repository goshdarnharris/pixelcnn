local BatchNarrow, parent = torch.class('nn.BatchNarrow', 'nn.Module')

-- Ported over from mainline torch. I'm using torch-cl, which doesn't yet have
-- a Narrow module that will support a negative dimension argument.

function BatchNarrow:__init(dimension,offset,length)
   parent.__init(self)
   self.dimension=dimension
   self.index=offset
   self.length=length or 1
   if not dimension or not offset then
      error('nn.Narrow(dimension, offset, length)')
   end
end

function BatchNarrow:updateOutput(input)
   local dim = self.dimension < 0 and input:dim() + self.dimension + 1 or self.dimension
   local length = self.length
   if length < 0 then
      length = input:size(dim) - self.index + self.length + 2
   end
   local output=input:narrow(dim,self.index,length)
   self.output = self.output:typeAs(output)
   self.output:resizeAs(output):copy(output)
   return self.output
end

function BatchNarrow:updateGradInput(input, gradOutput)
   local dim = self.dimension < 0 and input:dim() + self.dimension + 1 or self.dimension
   local length = self.length
   if length < 0 then
      length = input:size(dim) - self.index + self.length + 2
   end
   self.gradInput = self.gradInput:typeAs(input)
   self.gradInput:resizeAs(input):zero()
   self.gradInput:narrow(dim,self.index,length):copy(gradOutput)
   return self.gradInput
end

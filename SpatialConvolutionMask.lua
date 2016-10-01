local SpatialConvolutionMask, parent = torch.class('nn.SpatialConvolutionMask', 'nn.SpatialConvolution')

function SpatialConvolutionMask:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, mask)
   self.mask = mask
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)   
end

function SpatialConvolutionMask:reset(stdv)
   parent.reset(self, stdv)
   self.mask(self.weight)
end

function SpatialConvolutionMask:updateOutput(input)
   self.mask(self.weight)
   return parent.updateOutput(self, input)
end

require('torch')
require('optim')
require('nn')
require('nngraph')
require('GatedPixelConvolution')

local network = nn.Sequential()
network:add(GatedPixelConvolution(3, 8, 7, 1))
network:add(GatedPixelConvolution(8, 8, 3, 1))

-- local input = - nn.Identity()
-- local network = input - GatedPixelConvolution(3, 8, 7, 1) - GatedPixelConvolution(8, 8, 3, 1)
-- local network = nn.gModule({input}, {network})

print(network:forward({torch.Tensor(3,32,32), torch.Tensor(3,32,32)}))

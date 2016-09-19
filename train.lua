require('torch')
require('optim')
require('nn')
require('nngraph')
require('GatedPixelConvolution')
require('paths')
local gfx = require('display')


function infer(network, input)
	local output
	paths.rmall(network.name .. ".svg", "yes")
	local status, err = pcall(function() output = network:forward(input) end)
	if err then os.execute('open -a "/Applications/Google Chrome.app" ' .. network.name .. '.svg') end
	return output
end

nngraph.setDebug(true)

-- graph.dot(GatedPixelConvolution(3, 8, 7, 1).fg, 'gpc')



local model = GatedPixelConvolution(3, 8, 7, 1, 2)
model.name = "validate"
infer(model, {torch.Tensor(3,32,32), torch.Tensor(3,32,32)})

-- local network = nn.Sequential()
-- network:add(GatedPixelConvolution(3, 8, 7, 1))
-- network:add(GatedPixelConvolution(8, 8, 3, 1))

local input = - nn.Identity()
local network = input 
	- GatedPixelConvolution(3, 8, 7, 1, 1) 
	- GatedPixelConvolution(8, 8, 3, 1, 2)
	- GatedPixelConvolution(8, 16, 3, 1, 3)
	- GatedPixelConvolution(16, 16, 3, 1, 4)



local model = nn.gModule({input}, {network})
model.name = "pixelcnn"

print(string.format("model has %d parameters", model:getParameters():size()[1]))

-- print(model:forward({torch.Tensor(3,32,32), torch.Tensor(3,32,32)}))

local input = torch.rand(3,150,150)
local output = infer(model, {input,input})
gfx.image(output, {width=500})

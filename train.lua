require('torch')
require('optim')
require('nn')
require('nngraph')
require('pixelCNN')
require('paths')
optnet = require('optnet')
local gfx = require('display')

cmd = torch.CmdLine()
cmd:option("-dropout", .2, "dropout probability")
cmd:option("-epochs", 50, "number of training epochs")
cmd:option("-l2", 0, "l2 regularization strength")
cmd:option("-l1", 0, "l1 regularization strength")
cmd:option("-width", 100, "image width")
cmd:option("-height", 100, "image height")
cmd:option("-batchsize", 128, "batch size")
cmd:option("-cache", "cache", "image cache location")
cmd:option("-images", 1000, "number of images to use")
cmd:option("-embsize", false, "size of embedding vector")
cmd:option("-learningrate", 1e-2, "initial learning rate")
cmd:option("-learningratedecay", 4e-2, "learning rate decay per epoch")

opt = cmd:parse(arg)

opt.embsize = opt.embsize or (opt.width*opt.height/5)
gfx.verbose = false

function train(model, criterion, trainset, testset)
	params, gradParams = model:getParameters()
	local batchSize = opt.batchsize or 128
	local learningRate = opt.learningrate
	local learningRateDecay = opt.learningratedecay
	local optimState = {learningRate = opt.learningRate}
	local epoch

	local function do_loss(outputs, inputs)
		-- print(outputs:size())
		-- print(inputs:size())
		local loss = torch.Tensor(batchSize, opt.width, opt.height)
		local dloss_doutput = torch.Tensor(outputs:size())

		for x=1,opt.width do
			for y=1,opt.height do
				-- print(outputs[{{},{},x,y}]:size(), inputs[{{},x,y}]:size())
				local target = torch.floor(inputs[{{},{},x,y}]*255) + 1
				-- print(target)
				-- print(outputs:size())
				-- print(inputs:size())
				-- print(target)

				loss[{{},x,y}] = criterion:forward(outputs[{{},{},x,y}], target)
				dloss_doutput[{{},{},x,y}] = criterion:backward(outputs[{{},{},x,y}], target)
			end
		end

		return loss, dloss_doutput
	end

	-- Train a single pass through the inputs.
	local function step()
		model:training()

		local total_loss = 0		
		local shuffle = torch.randperm(trainset.size)

		for batchIndex=1,trainset.size,batchSize do
			-- io.write(string.format("\rconstructing batch %d - %d", batchIndex, batchIndex + batchSize))
			-- io.flush()
			-- Create the batch
			local size = math.min(batchIndex + batchSize, trainset.size) - batchIndex
			local batchInputs = torch.Tensor(size, 1, opt.width, opt.height)

			for j=1,size do
				local idx = shuffle[batchIndex+j]
				local input = torch.Tensor(1,opt.width,opt.height)
				input:zero()
				input[{1,{},{}}] = trainset.data[idx]:select(1,3)

				batchInputs[j] = input
			end

			-- Train the batch
			local function feval(params)
				gradParams:zero()

				-- Evaluating this model consists of:
				-- a forward pass with the target image as input
				-- a forward pass of the criterion for each pixel with the 256-long slice of log likelihoods (one for each possible value)
				-- a backward pass of the criterion for each of those
				-- reassemble losses and dloss_doutput into the full image (N x M x 256, where each
				-- [n][m] index is a single criterion pass above)
				-- backprop that through the network

				-- Oy.

				local outputs = model:forward(batchInputs)
				-- print(#outputs)
				local loss, dloss_doutput = do_loss(outputs, batchInputs)
				model:backward(batchInputs, dloss_doutput)

				-- loss = loss + opt.l2 * torch.norm(params,2)^2/2
				-- loss = loss + opt.l1 * torch.norm(params,1)

				return loss, gradParams
			end

			_, result = optim.adagrad(feval, params, optimState)
			total_loss = total_loss + result[1]:mean()

			io.write(string.format("\rT: epoch %d: batch %d of %d: loss %.4f [mean %.4f]", epoch, math.ceil(batchIndex/size), math.ceil(trainset.size/size), result[1]:mean(), total_loss/(batchIndex/size + 1)))
			io.flush()
		end

		-- Return normalized loss
		return total_loss/math.ceil(trainset.size/batchSize)
	end

	-- Evaluate the model against the test set.
	local function test()
		model:evaluate()

		local total_loss = 0
		local accuracy = 0
		for i=1,testset.size do
			local input = torch.Tensor(1,1,opt.width,opt.height)
			input:zero()
			input[{1,1,{},{}}] = testset.data[i]:select(1,3)
			local output = model:forward(input)

			local loss,_ = do_loss(output, input)
			-- total_loss = total_loss + criterion:forward(output, input):mean()
			total_loss = loss:mean() + total_loss
			local _,index = torch.max(output, 1)
		end

		return total_loss/testset.size
	end


	for i=1,opt.epochs do
		epoch = i
		local train_loss = step()
		local val_loss, accuracy = test()
		print(string.format("\nV: epoch %d: training loss %.4f, validation loss %.4f [rate = %f]", i, train_loss, val_loss, learningRate))

		-- FIXME: this is super ugly and needs to be cleaned up.
		-- Which means that the pixelCNN needs to be modified to do multiple channels,
		-- and properly select slices of the input when too many channels are provided as input. Maybe.
		local input = testset.data[torch.random(1,testset.size)]
		local output = model:forward(input)
		local img = pixelCNN.toImage(output)

		-- print(input[1]:size(), img[1]:size())
		gfx.image({image.hsv2rgb(source), image.hsv2rgb(img)}, {width = 300, win = "asdf"})
		learningRate = learningRate/(1+learningRateDecay)
		torch.save(string.format("cv/model-%d-%.3f-%.3f.t7", i, train_loss, val_loss))
	end

end

function infer(network, input)
	local output
	paths.rmall(network.name .. ".svg", "yes")

	local go = function()
		output = network:forward(input)
	end

	local status, err = pcall(go)
	if err then os.execute('open -a "/Applications/Google Chrome.app" ' .. network.name .. '.svg') end
	return output
end

function graphgen(network, input, name)
	name = name or network.name
	local generateGraph = require 'optnet.graphgen'

	graphOpts = {
		displayProps =  {shape='ellipse',fontsize=14, style='solid'},
		nodeData = function(oldData, tensor)
			return oldData .. '\n' .. 'Size: '.. tensor:numel()
		end
	}

	local g = generateGraph(network, input, graphOpts)
	graph.dot(g, name, name)
end

print("----- Creating model.")
nngraph.setDebug(true)

local model = nn.Sequential()
model:add(pixelCNN.inputLayer(8, 7, 1))
model:add(pixelCNN.GatedPixelConvolution(8, 16, 3, 1))


model.name = "validate"
print(infer(model, torch.Tensor(10,3,32,32))[1]:size())

print(infer(model, torch.Tensor(3,32,32))[1]:size())


local helper = pixelCNN.Helper()
helper:addLayer(16, 7)
helper:addLayer(32, 3)
helper:addLayer(32, 3)
helper:addLayer(32, 3)
helper:addLayer(32, 3)
helper:addLayer(32, 3)

local model = helper:generate("test")

-- print(infer(model, torch.Tensor(3,32,32)))
print(model:forward(torch.Tensor(3,32,32)))

-- TODO: visualize the network after running to ensure that it's doing the job
-- properly. May be easiest to make it shit itself at the very last step.

-- local input = - nn.Replicate(2)
-- local network = input 
-- 	- nn.SplitTable(1)
-- 	- pixelCNN.GatedPixelConvolution(1, 64, 7, 1, 1) 
-- 	- pixelCNN.GatedPixelConvolution(64, 64, 5, 1, 2)
-- 	- pixelCNN.GatedPixelConvolution(64, 256, 5, 1, 3)
-- 	- pixelCNN.GatedPixelConvolution(256, 256, 5, 1, 4)
-- 	- nn.SelectTable(2)
-- 	- pixelCNN.MultiChannelSpatialSoftMax(1)

-- -- TODO: input pre-processing
-- -- the paper uses centering and scaling; that's it.
-- -- play with ZCA and HSV as well.
-- -- paper uses RMSProp for optimization.
-- -- nll loss function
-- -- batch normalization?

-- local model = nn.gModule({input}, {network})
-- model.name = "pixelcnn"

-- print(string.format("model has %d parameters", model:getParameters():size()[1]))

-- -- print(model:forward({torch.Tensor(3,32,32), torch.Tensor(3,32,32)}))

-- local input = torch.rand(5,1,32,32)

-- -- graphgen(model, input, "unoptimized")
-- local time = torch.Timer()
-- local out = infer(model, input)
-- print("model takes " .. time:time().real .. "s")
-- print("output dims are", out:size())

-- -- gfx.image(pixelCNN.toImage({out}))
-- print("Model valid.")

-- print("----- Loading dataset.")
-- print(string.format("cache location is %s", opt.cache))

-- require('data')
-- local trainset, testset = data.load("data/tgif-v1.0.tsv", opt.width, opt.height, opt.images, .9, opt.cache)
-- print(string.format("training set has %d images", trainset.size))
-- print(string.format("test set has %d images", testset.size))

-- local criterion = nn.CrossEntropyCriterion()

-- print("----- Starting training.")
-- train(model, criterion, trainset, testset)

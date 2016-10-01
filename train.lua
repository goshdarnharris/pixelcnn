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
cmd:option("-learningrate", 1e-1, "initial learning rate")
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
	local channels = 3
	local epoch

	local function do_loss(outputs, inputs)
		local loss = {}
		local dloss_doutput = {}

		for channel=1,channels do
			if inputs:dim() > 3 then
				loss[channel] = torch.Tensor(batchSize, opt.width, opt.height)
				dloss_doutput[channel] = torch.Tensor(outputs[channel]:size())

				for x=1,opt.width do
					for y=1,opt.height do
						local classes = outputs[channel][1]:size()[1]
						local target = torch.floor(inputs[{{},channel,x,y}]*(classes-1)) + 1
						local out = outputs[channel][{{},{},x,y}]
						loss[channel][{{},x,y}] = criterion:forward(out, target)
						dloss_doutput[channel][{{},{},x,y}] = criterion:backward(out, target)
					end
				end
			else
				loss[channel] = torch.Tensor(opt.width, opt.height)
				dloss_doutput[channel] = torch.Tensor(outputs[channel]:size())

				for x=1,opt.width do
					for y=1,opt.height do
						local classes = outputs[channel]:size()[1]
						local target = torch.floor(inputs[{channel,x,y}]*(classes-1)) + 1
						local out = outputs[channel][{{},x,y}]
						loss[channel][{x,y}] = criterion:forward(out, target)
						dloss_doutput[channel][{{},x,y}] = criterion:backward(out, target)
					end
				end
			end
		end

		return loss, dloss_doutput
	end

	local function mean_loss(loss)
		local mean = 0
		for channel=1,channels do
			mean = mean + loss[channel]:mean()/channels
		end
		return mean
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
			local batchInputs = torch.Tensor(size, 3, opt.width, opt.height)

			for j=1,size do
				local idx = shuffle[batchIndex+j]
				local input = image.hsv2rgb(trainset.data[idx])

				batchInputs[j] = input
			end

			-- Train the batch
			local function feval(params)
				gradParams:zero()

				-- Evaluating this model consists of:
				-- a forward pass with the target image as input
				-- a forward pass of the criterion for each pixel with the <classes>-long slice of log likelihoods (one for each possible value)
				-- a backward pass of the criterion for each of those
				-- reassemble losses and dloss_doutput into the full image (N x M x classes, where each
				-- [n][m] index is a single criterion pass above)
				-- backprop that through the network

				local outputs = model:forward(batchInputs)
				local loss, dloss_doutput = do_loss(outputs, batchInputs)
				model:backward(batchInputs, dloss_doutput)

				return loss, gradParams
			end

			_, result = optim.adagrad(feval, params, optimState)
			total_loss = total_loss + mean_loss(result[1])

			io.write(string.format("\rT: epoch %d: batch %d of %d: loss %.4f [mean %.4f]", epoch, math.ceil(batchIndex/batchSize), math.ceil(trainset.size/batchSize), mean_loss(result[1]), total_loss/(batchIndex/batchSize + 1)))
			io.flush()
		end

		-- Return normalized loss
		return total_loss/math.ceil(trainset.size/batchSize)
	end

	-- Evaluate the model against the test set.
	local function validate()
		model:evaluate()

		local total_loss = 0
		local accuracy = 0
		for i=1,testset.size do
			local input = image.hsv2rgb(testset.data[i])
			local output = model:forward(input)

			local loss,_ = do_loss(output, input)
			total_loss = mean_loss(result[1]) + total_loss			
		end

		return total_loss/testset.size
	end

	local losses = {}
	for i=1,opt.epochs do
		epoch = i
		local train_loss = step()
		local val_loss, accuracy = validate()
		print(string.format("\nV: epoch %d: training loss %.4f, validation loss %.4f [rate = %f]", i, train_loss, val_loss, learningRate))

		local input = image.hsv2rgb(testset.data[torch.random(1,testset.size)])
		local output = model:forward(input)
		local img = pixelCNN.toImage(output)

		-- print(input)
		-- print(output[1][{{},{},1}])
		losses[epoch] = {epoch, val_loss}

		gfx.image({input, img}, {width = 500, win = "test output"})
		gfx.plot(losses, {win = "losses", xlabel = "epoch", ylabel = "val loss", title = "losses"})

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


local helper = pixelCNN.Helper()
helper:addLayer(128, 7)
helper:addLayer(128, 3)
helper:addLayer(128, 3)
helper:addLayer(128, 3)

local model = helper:generate("pixelcnn")

-- print(infer(model, torch.Tensor(3,32,32)))
print(model:forward(torch.Tensor(3,32,32)))

-- TODO: input pre-processing
-- the paper uses centering and scaling; that's it.
-- play with ZCA and HSV as well.
-- paper uses RMSProp for optimization.
-- nll loss function
-- batch normalization?


print(string.format("model has %d parameters", model:getParameters():size()[1]))

-- print(model:forward({torch.Tensor(3,32,32), torch.Tensor(3,32,32)}))


-- gfx.image(pixelCNN.toImage({out}))
print("Model valid.")

print("----- Loading dataset.")
print(string.format("cache location is %s", opt.cache))

require('data')
local trainset, testset = data.load("data/tgif-v1.0.tsv", opt.width, opt.height, opt.images, .9, opt.cache)
print(string.format("training set has %d images", trainset.size))
print(string.format("test set has %d images", testset.size))

local criterion = nn.CrossEntropyCriterion()

print("----- Starting training.")
torch.setnumthreads(4)
train(model, criterion, trainset, testset)

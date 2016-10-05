-- A convolution layer with a gated activation unit as described in the Conditional PixelCNN paper.
-- This convolution layer includes a vertical/horizontal stack, and is intended for use in image generation.

require('torch')
require('nn')
require('nngraph')
require('SpatialConvolutionMask')
require('BatchNarrow')
require('BiasTable')

-- local GatedPixelConvolution = torch.class('GatedPixelConvolution')
pixelCNN = {}

function pixelCNN.toImage(output)
	-- Expects raw output from the network, which is 
	-- each table index is a channel of the output image.
	local dim = output[1]:dim()
	local size = output[1]:size()
	local batched = false
	local image
	local width
	local height
	local classes
	-- print(size)

	if dim > 4 then
		error("output channels must have at most 4 dimensions (batch x classes x N x M)")
	elseif dim < 3 then
		error("output channels must have at least 3 dimensions (classes x N x M)")
	end

	if dim == 4 then batched = true end

	if batched then
		classes = size[2]
		width = size[3]
		height = size[4]
		image = torch.Tensor(size[1], 3, width, height)
		image:fill(1)
	else
		classes = size[1]
		width = size[2]
		height = size[3]
		image = torch.Tensor(3, width, height)
		image:fill(1)
	end

	local function getValues(channel)
		local size = channel:size()
		local values = torch.Tensor(width, height)

		for x=1,width do
			for y=1,height do
				local value, _
				_,value = channel[{{},x,y}]:max(1)
				-- print(channel[{{},x,y}])
				-- print(value)
				values[{x,y}] = value[1]/classes
			end
		end

		return values
	end

	for idx,channel in pairs(output) do
		if not channel:isSize(size) then
			error("all output channels must have the same size")
		end

		if batched then
			for batchIdx = 1,size[1] do
				-- print(image[{batchIdx,idx,{},{}}]:size())
				image[{batchIdx,idx,{},{}}] = getValues(channel[batchIdx])
			end
		else
			image[{idx,{},{}}] = getValues(channel)
		end
	end

	-- if #output < 3 then
	-- 	for idx=#output+1,3 do
	-- 		image[idx]

	return image
end

function pixelCNN.GAU(nInputPlane, left, right)
	left_type = left or nn.Tanh
	right_type = right or nn.Sigmoid

	local input = - nn.Identity()

	local left = input 
		- nn.BatchNarrow(-3, 1, nInputPlane/2)
		- left_type()

	local right = input
		- nn.BatchNarrow(-3, nInputPlane/2, nInputPlane/2)
		- right_type()

	local gate = nn.CMulTable()({left,right})

	return nn.gModule({input}, {gate})
end

function pixelCNN.inputLayer(nOutputPlane, kernel_size, kernel_step, embedding_size, channels)
	channels = channels or 3

	local input = nn.ConcatTable()

	-- Input to vertical stack is just the input image
	input:add(nn.SelectTable(1))

	-- Input to horiz is a copy
	local copy = nn.Sequential()
	copy:add(nn.SelectTable(1))
	copy:add(nn.Copy(torch.Tensor.__typename, torch.Tensor.__typename, true, true))

	input:add(copy)

	-- Embedding can go right through
	input:add(nn.SelectTable(2))

	-- Convert to a node
	input = input()

	local layer = input
		- pixelCNN.GatedPixelConvolution(1, nOutputPlane, kernel_size, kernel_step, embedding_size, 1, channels, false)
	layer:annotate{name = 'input layer', description = 'massage input, first pixel convolution layer'}

	return nn.gModule({input}, {layer})
end

function pixelCNN.outputLayer(nInputPlane, channels)
	channels = channels or 3

	local input = nn.SelectTable(2)():annotate{
		description = 'select horiz stack output'
	}
	local outputs = {}

	for channel=1,channels do
		local start = (channel - 1)*nInputPlane + 1
		outputs[channel] = nn.BatchNarrow(-3, start, nInputPlane)(input):annotate{
			name = 'out_ch' .. channel,
			description = 'split output for channel ' .. channel
		}

		outputs[channel] = nn.SpatialSoftMax()(outputs[channel]):annotate{
			name = 'softmax_ch' .. channel,
			description = 'softmax for channel ' .. channel
		}
	end

	return nn.gModule({input}, outputs)
end

function pixelCNN.GatedPixelConvolution(nInputPlane, nOutputPlane, kernel_size, kernel_step, embedding_size, layer, channels, force_residual)
	-- nInputPlane and nOutputPlane are per-channel. This will multiply accordingly.
	
	-- Force residuals by default
	if force_residual == undefined then force_residual = true end
	-- If this is the first layer, disable residuals so information about the current pixel doesn't get through.
	if layer == 1 then force_residual = false end

	layer = layer or 3
	channels = channels or 3

	-- If we're at the first layer, we can't look at the current pixel.
	-- If we're at a later layer, we can.
	-- If we're at the second layer, a channel doesn't look at itself as an input.
	-- If we're at a later layer, it does.

	-- In each layer, there is one full vertical & horizontal stack per channel. The layer takes in
	-- N x M x channels x nInputPlane features.
	-- Beyond the 2nd layer, each channel takes every channel up to and including itself as
	-- and input, and each channel produces an N x M x nOutputPlane output.
	-- At the second layer, this is the same except each channel does not take its corresponding input.
	-- At the first layer, the input is N x M x channels in size.

	-- It may be simpler to just have 3 parallel convolutional stacks, and concatenate their
	-- outputs. I -think- that's entirely equivalent, though that depends on what the gated
	-- unit does (well, it has no weights)
	-- That is, where there are currently single convolutions, I can maybe place concat layers to handle it
	-- Or, I just put 3 of these in parallel with the appropriate # of planes, then narrow and concat as appropriate.

	-- TODO: implement dilated convolutions a la wavenet

	local hstack_in, hstack_out
	local vstack_in, vstack_out
	local embedding_input, embedding_transform
	-- Each stack has a convolution that outputs 2*nOutputPlane features. These are split
	-- at the gate; the first nOutputPlane features go to the left gate and the second
	-- nOutputPlane features go to the right gate.

	-- If no embedding is used, this just forward the third input along to the next layer.
	embedding_input = nn.Identity()():annotate{
		name = 'emb_in',
		description = 'pass through to forward embedding to next layer'
	}

	if embedding_size > 0 then
		-- Create the linear transformation for the embedding. This will be added to
		-- the convolution features as a bias immediately before they are gated.
		embedding_transform = nn.Linear(embedding_size, 2*nOutputPlane*channels)(embedding_input):annotate{
			name = 'emb',
			description = 'transform embedding input'
		}
	end


	-- Create vertical padding, convolution, and crop
	local vertical_conv, vertical_pad
	do
		local kW = kernel_size
		local kH = math.floor(kernel_size/2) -- Paper says ceil. There are only floor rows above the current pixel...
		local dW = kernel_step
		local dH = kernel_step
		local padW = 0
		local padH = 0

		vertical_pad = nn.SpatialZeroPadding(
			math.floor(kernel_size/2), 
			math.floor(kernel_size/2),
			math.floor(kernel_size/2),
			-1)
		vertical_conv = nn.SpatialConvolution(nInputPlane*channels, 2*nOutputPlane*channels, kW, kH, dW, dH, padW, padH)
		local params,_ = vertical_conv:getParameters()
	end

	-- Create horizontal padding, convolution, and crop
	local horiz_conv, horiz_pad
	do
		local kH = 1
		local dW = kernel_step
		local dH = kernel_step
		local padW = 0
		local padH = 0

		local kW
		if layer == 1 then
			kW = math.floor(kernel_size/2) -- Don't include the current pixel
		else
			kW = math.ceil(kernel_size/2) -- Include the current pixel
		end

		

		if layer == 1 then
			-- Sizes mean we'll still have 1 extra pixel on the right side of the image. Get rid of it.
			horiz_pad = nn.SpatialZeroPadding(math.floor(kernel_size/2), -1, 0, 0)
			-- We do a normal convolution with no mask; the current pixel is fully masked by the kernel width.
			horiz_conv = nn.SpatialConvolution(nInputPlane*channels, 2*nOutputPlane*channels, kW, kH, dW, dH, padW, padH)
		else
			-- Since other layers use a wider convolution, the extra pixel will already be gone.
			horiz_pad = nn.SpatialZeroPadding(math.floor(kernel_size/2), 0, 0, 0)
			
			-- local test = torch.Tensor(2*nOutputPlane*channels, nInputPlane*channels, kH, kW):fill(1)

			-- But we need to mask the current pixel.
			-- Each channel can only see itself and the preceding channels. All other channels in 
			-- the kernels for a channel's output planes must be masked out.
			function mask(weights)
				for channel=1,channels-1 do
					local start = nOutputPlane*(channel - 1) + 1
					local stop = nOutputPlane*channel

					weights[{{start, stop}, {nInputPlane*channel+1,nInputPlane*channels}, {}, kW}] = 0

					-- Since there are two sets of outputs (one for each gate in the GAU), we need
					-- to duplicate the first half of the mask into the second half
					start = start + nOutputPlane*channels
					stop = stop + nOutputPlane*channels
					weights[{{start, stop}, {nInputPlane*channel+1,nInputPlane*channels}, {}, kW}] = 0
				end
			end

			horiz_conv = nn.SpatialConvolutionMask(nInputPlane*channels, 2*nOutputPlane*channels, kW, kH, dW, dH, padW, padH, mask)
		end
	end

	vstack_in = vertical_pad():annotate{
		name = 'vpad', 
		description = 'padding for vertical convolution',
	}

	local vconv_out = vertical_conv(vstack_in):annotate{
		name = 'vconv', 
		description = vertical_conv.kW .. 'x' .. vertical_conv.kH .. ' vertical convolution',
	}

	if embedding_size > 0 then
		vconv_out = nn.BiasTable()({vconv_out, embedding_transform}):annotate{
			name = 'vbias',
			description = 'apply embedding bias to vertical stack'
		}
	end

	vstack_out = pixelCNN.GAU(2*nOutputPlane*channels)(vconv_out):annotate{
		name = 'vgate',
		description = 'gated output for vertical stack',
	}  -- Output is gated vertical convolution


	hstack_in = nn.Identity()()

	local hconv_out = horiz_pad(hstack_in):annotate{
		name = 'hpad', 
		description = 'padding for horiz convolution',
	}

	hconv_out = horiz_conv(hconv_out):annotate{
		name = 'hconv', 
		description = horiz_conv.kW .. 'x' .. horiz_conv.kH .. ' horiz convolution',
	}

	local vtohconv = nn.SpatialConvolution(2*nOutputPlane*channels, 2*nOutputPlane*channels, 1, 1, 1, 1)
	local vconv_to_hconv = vtohconv(vconv_out):annotate{
		name = 'vtohconv',
		description = vtohconv.kW .. 'x' .. vtohconv.kH .. ' vertical conv -> horiz stack conv',
	}


	local hstack = nn.CAddTable()({hconv_out, vconv_to_hconv}):annotate{
		name = 'vhadd',
		description = 'add vertical convolution to horiz convolution',
	}

	if embedding_size > 0 then
		hstack = nn.BiasTable()({hstack, embedding_transform}):annotate{
			name = 'hbias',
			description = 'apply embedding bias to horiz stack'
		}
	end

	hstack = pixelCNN.GAU(2*nOutputPlane*channels)(hstack):annotate{
		name = 'hgate',
		description = 'gated output for horiz stack',
	}

	local hstack_outconv = nn.SpatialConvolution(nOutputPlane*channels, nOutputPlane*channels, 1, 1, 1, 1)
	hstack = hstack_outconv(hstack):annotate{
		name = 'hstackconv',
		description = hstack_outconv.kW .. 'x' .. hstack_outconv.kH .. ' horiz stack output conv',
	}

	-- hstack output is gated elementwise addition of vconv output (convolved w/ 1x1 filter) and hconv output

	-- Add residual on horizontal stack
	
	if force_residual then
		if nInputPlane == nOutputPlane then
			-- If input and output dimensions are the same, we can just use the identity.
			hstack_out = nn.CAddTable()({hstack_in, hstack}):annotate{
				name = 'hstack_residual',
				description = 'horiz stack residual',
			}
		else
			-- Otherwise, use a 1x1 convolution to transform dimensions.
			local residual_conv = nn.SpatialConvolution(nInputPlane*channels, nOutputPlane*channels, 1, 1, 1, 1)

			local transform = residual_conv(hstack_in):annotate{
				name = 'hstack_residualconv',
				description = residual_conv.kW .. 'x' .. residual_conv.kH .. ' horiz stack residual transform',
			}
			hstack_out = nn.CAddTable()({transform, hstack}):annotate{
				name = 'hstack_residual',
				description = 'horiz stack residual',
			}
		end
	else
		-- hstack output is directly from the stack. no residual.
		hstack_out = hstack
	end

	nngraph.annotateNodes()

	-- FIXME: is this correct? the paper mentions combining the outputs after each layer, but also
	-- mentions that combining the horizontal stack with the vertical stack would allow the vertical
	-- stack to see future pixels. Hmm.

	-- Each layer takes the previous layer's vertical and horizontal stacks as well as any embedding input.
	-- The embedding input is used internally and forwarded unchanged to the next layer.
	return nn.gModule({vstack_in, hstack_in, embedding_input}, {vstack_out, hstack_out, embedding_input})
end

local Helper = torch.class('pixelCNN.Helper')

function Helper:__init(opts)
	opts = opts or {}
	self.channels = opts.channels or 3
	self.force_residual = opts.force_residual or true
	self.embedding_size = opts.embedding_size or 0
	self.sample_input = opts.input or {torch.Tensor(3,32,32), torch.Tensor(self.embedding_size)}

	self.layers = {}
end

function Helper:addLayer(nOutputPlane, kernel_size)

	if #self.layers == 0 then
		-- Create the input layer
		self.layers[#self.layers+1] = {
			layer = pixelCNN.inputLayer(nOutputPlane, kernel_size, 1, self.embedding_size, self.channels),
			nOutputPlane = nOutputPlane
		}
	else
		self.layers[#self.layers+1] = {
			layer = pixelCNN.GatedPixelConvolution(
				self.layers[#self.layers].nOutputPlane, nOutputPlane,
				kernel_size, 1, self.embedding_size, #self.layers+1, self.channels, self.force_residual),
			nOutputPlane = nOutputPlane
		}
	end

	self.layers[#self.layers].name = "pixelCNN_l" .. #self.layers
end

function Helper:generate(name)
	-- Add output layer
	local output = pixelCNN.outputLayer(self.layers[#self.layers].nOutputPlane, self.channels)
	local model = nn.Sequential()
	model.name = name

	local function check(model, layer, idx)
		-- Try to run output through it so we can better debug the final assembly.
		local status, err = pcall(function() model:forward(self.sample_input) end)
		if err then
			print("pixelCNN generation failed at layer " .. idx .. ": " .. tostring(layer.layer))
			error(err)
		end
	end

	-- Do it to it
	for idx,layer in pairs(self.layers) do
		model:add(layer.layer)
		check(model, layer, idx)		
	end
	model:add(output)
	check(model, output, #self.layers + 1)

	return model
end



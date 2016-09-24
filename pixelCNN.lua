-- A convolution layer with a gated activation unit as described in the Conditional PixelCNN paper.
-- This convolution layer includes a vertical/horizontal stack, and is intended for use in image generation.

require('torch')
require('nn')
require('nngraph')

-- local GatedPixelConvolution = torch.class('GatedPixelConvolution')

local function toImage(output)
	-- Output should be a table of batchSize x 256 x N x M tensors
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
		-- print(values:size())
		for x=1,width do
			for y=1,height do
				local value, _
				_,value = channel[{{},x,y}]:max(1)
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
			image[channel] = getValues(channel)
		end
	end

	-- if #output < 3 then
	-- 	for idx=#output+1,3 do
	-- 		image[idx]

	return image
end

local function GAU(planes, left, right)
	left_type = left or nn.Tanh
	right_type = right or nn.Sigmoid

	local input = - nn.Identity()

	local left = input 
		- nn.Narrow(2, 1, planes/2)
		- left_type()

	local right = input
		- nn.Narrow(2, planes/2, planes/2)
		- right_type()

	local gate = nn.CMulTable()({left,right})

	return nn.gModule({input}, {gate})
end

local function MultiChannelSpatialSoftMax(channels)
	if channels == 1 then
		-- Easy degenerate case.
		return nn.SpatialSoftMax()
	end
end

local function GatedPixelConvolution(nInputPlane, nOutputPlane, kernel_size, kernel_step, layer, channels, residual)
	-- nInputPlane and nOutputPlane are per-channel.

	residual = residual or true
	layer = layer or 3
	channels = channels or 1

	-- If we're at the first layer, we can't look at the current pixel.
	-- If we're at a later layer, we can.
	-- If we're at the second layer, a channel doesn't look at itself as an input.
	-- If we're at a later layer, it does.

	-- In each layer, there is one full convolutional unit per channel. The layer takes in
	-- N x M x channels x nInputPlane features.
	-- Beyond the 2nd layer, each channel takes every channel up to and including itself as
	-- and input, and each channel produces an N x M x nOutputPlane output.
	-- At the second layer, this is the same except each channel does not take its corresponding input.
	-- At the first layer, the input is N x M x channels in size.

	-- If nInputPlane and nOutputPlane are the same, force residuals.
	if nInputPlane == nOutputPlane then residual = true end

	-- TODO: implement dilated convolutions a la wavenet

	-- Vertical stack can only depend on pixels above the current pixel. This can be achieved by 
	-- using a filter of half height, and padding the input appropriately.

	local vertical_conv, vertical_pad
	do
		local kW = kernel_size
		local kH = math.floor(kernel_size/2) -- Paper says ceil. I don't believe it.
		local dW = kernel_step
		local dH = kernel_step
		local padW = 0
		local padH = 0

		vertical_pad = nn.SpatialZeroPadding(
			math.floor(kernel_size/2), 
			math.floor(kernel_size/2),
			math.floor(kernel_size/2),
			0)
		vertical_conv = nn.SpatialConvolution(nInputPlane, 2*nOutputPlane, kW, kH, dW, dH, padW, padH)
		local params,_ = vertical_conv:getParameters()
		-- print("vertical convolution has " .. params:storage():size() .. " hidden units")
		-- print("should be " .. kW*kH*nInputPlane*nOutputPlane)
		-- We'll have 1 extra pixel on the bottom of the image. Get rid of it.
		vertical_crop = nn.SpatialZeroPadding(0, 0, 0, -1)
	end

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

		horiz_pad = nn.SpatialZeroPadding(math.floor(kernel_size/2), 0, 0, 0)
		-- nn.Padding(3, -math.floor(kernel_size/2), 3)
		horiz_conv = nn.SpatialConvolution(nInputPlane, 2*nOutputPlane, kW, kH, dW, dH, padW, padH)
		local params,_ = horiz_conv:getParameters()
		-- print("horizontal convolution has " .. params:storage():size() .. " hidden units")

		if layer == 1 then
			-- Sizes mean we'll still have 1 extra pixel on the right side of the image. Get rid of it.
			-- horiz_crop = nn.Narrow(3, 1, -2)
			horiz_crop = nn.SpatialZeroPadding(0, -1, 0, 0)
		else
			horiz_crop = nn.Identity()
		end
	end

	-- Input to the layer is {tensor, tensor}. The first tensor goes to the vertical stack; the second to the
	-- horizontal stack.

	local vstack_in = - vertical_pad -- Output is nxn vertical convolution
	local vconv_out = vstack_in - vertical_conv - vertical_crop
	local vstack_out = vconv_out - GAU(2*nOutputPlane) -- Output is gated vertical convolution

	local hstack_in = - nn.Identity()
	local hconv_out = hstack_in - horiz_pad - horiz_conv - horiz_crop -- Output is 1xn horizontal convolution

	local vconv_to_hconv = vconv_out - nn.SpatialConvolution(2*nOutputPlane, 2*nOutputPlane, 1, 1, 1, 1)

	local hstack = GAU(2*nOutputPlane)(nn.CAddTable()({hconv_out, vconv_to_hconv})) 
		- nn.SpatialConvolution(nOutputPlane, nOutputPlane, 1, 1, 1, 1)
	-- hstack output is gated elementwise addition of vconv output (convolved w/ 1x1 filter) and hconv output

	-- Add residual on horizontal stack
	local hstack_out
	if residual then
		if nInputPlane == nOutputPlane then
			-- If input and output dimensions are the same, we can just use the identity.
			hstack_out = nn.CAddTable()({hstack_in, hstack})
		else
			-- Otherwise, use a 1x1 convolution to transform dimensions.
			local transform = hstack_in - nn.SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, 1, 1)
			hstack_out = nn.CAddTable()({transform, hstack})
		end
	else
		-- hstack output is directly from the stack. no residual.
		hstack_out = hstack
	end

	nngraph.annotateNodes()

	return nn.gModule({vstack_in, hstack_in}, {vstack_out, hstack_out})
end

pixelCNN = {
	toImage = toImage,
	MultiChannelSpatialSoftMax = MultiChannelSpatialSoftMax,
	GAU = GAU,
	GatedPixelConvolution = GatedPixelConvolution
}


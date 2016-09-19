-- A convolution layer with a gated activation unit as described in the Conditional PixelCNN paper.
-- This convolution layer includes a vertical/horizontal stack, and is intended for use in image generation.

require('torch')
require('nn')
require('nngraph')

-- local GatedPixelConvolution = torch.class('GatedPixelConvolution')

function GAU(planes, left, right)
	left_type = left or nn.Tanh
	right_type = right or nn.Sigmoid

	local input = - nn.Identity()

	local left = input 
		- nn.Narrow(1, 1, planes/2)
		- left_type()

	local right = input
		- nn.Narrow(1, planes/2, planes/2)
		- right_type()

	local gate = nn.CMulTable()({left,right})

	return nn.gModule({input}, {gate})
end

function GatedPixelConvolution(nInputPlane, nOutputPlane, kernel_size, kernel_step, layer, channels, residual)
	-- nInputPlane and nOutputPlane are per-channel.
	
	residual = residual or false
	layer = layer or 3
	channels = channels or 1

	-- If we're at the first layer, we can't look at the current pixel.
	-- If we're at a later layer, we can.
	-- If we're at the second layer, a channel doesn't look at itself as an input.
	-- If we're at a later layer, it does.

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

		vertical_pad = nn.Sequential()
		vertical_pad:add(nn.Padding(2, -math.floor(kernel_size/2), 3))
		vertical_pad:add(nn.Padding(3, -math.floor(kernel_size/2), 3)) -- Paper says ceil. I don't believe it.
		vertical_pad:add(nn.Padding(3, math.floor(kernel_size/2), 3))
		vertical_conv = nn.SpatialConvolution(nInputPlane, 2*nOutputPlane, kW, kH, dW, dH, padW, padH)
		-- We'll have 1 extra pixel on the bottom of the image. Get rid of it.
		vertical_crop = nn.Narrow(2, 1, -2)
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

		horiz_pad = nn.Padding(3, -math.floor(kernel_size/2), 3)
		horiz_conv = nn.SpatialConvolution(nInputPlane, 2*nOutputPlane, kW, kH, dW, dH, padW, padH)

		if layer == 1 then
			-- Sizes mean we'll still have 1 extra pixel on the right side of the image. Get rid of it.
			horiz_crop = nn.Narrow(3, 1, -2)
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


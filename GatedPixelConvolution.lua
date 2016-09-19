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

function GatedPixelConvolution(nInputPlane, nOutputPlane, kernel_size, kernel_step)
	-- TODO: implement dilated convolutions a la wavenet

	-- Vertical stack can only depend on pixels above the current pixel. This can be achieved by 
	-- using a filter of half height, and padding the input appropriately.

	-- The paper says to use ceil(n/2) x n and ceil(n/2) x 1 convolutions for the horizontal and
	-- vertical stacks, respectively. I think they mean floor(n/2) x n and floor(n/2) x 1.

	local vertical_conv
	do
		local kW = kernel_size
		local kH = math.floor(kernel_size/2) -- Paper says ceil. I don't believe it.
		local dW = kernel_step
		local dH = kernel_step
		local padW = math.floor(kernel_size/2)
		local padH = math.floor(kernel_size/2) -- Paper says ceil. I don't believe it.
		vertical_conv = nn.SpatialConvolution(nInputPlane, 2*nOutputPlane, kW, kH, dW, dH, padW, padH)

		-- For an NxN input, the vertical convolution will produce an output of
		-- (N + 2*floor(kernel_size/2) - kernel_size)/kernel_step + 1
		-- by
		-- (N + floor(kernel_size/2))/kernel_step + 1
		-- for a 3x3 kernel w/ step 1, the output will be N x N+2.
		-- This means that the output of this convolution needs to be cropped by
		-- floor(kernel_size/2) + 1 pixels. These pixels should be cropped from the
		-- bottom of the image.
	end

	local horiz_conv
	do
		local kW = math.floor(kernel_size/2) -- Paper says ceil.
		local kH = 1
		local dW = kernel_step
		local dH = kernel_step
		local padW = math.floor(kernel_size/2)
		local padH = 0
		horiz_conv = nn.SpatialConvolution(nInputPlane, 2*nOutputPlane, kW, kH, dW, dH, padW, padH)

		-- Similar to the vertical convolution, the output of the horizonal convolution will need to be cropped.
		-- They should be cropped from the right side of the image.
	end

	-- Input to the layer is {tensor, tensor}. The first tensor goes to the vertical stack; the second to the
	-- horizontal stack.

	local vstack_in = - vertical_conv -- Output is nxn vertical convolution
	local vstack_out = vstack_in - GAU(2*nOutputPlane) -- Output is gated vertical convolution

	local hstack_in = - nn.Identity()
	local hconv_out = hstack_in - horiz_conv -- Output is 1xn horizontal convolution

	local vconv_to_hconv = vstack_in - nn.SpatialConvolution(2*nOutputPlane, 2*nOutputPlane, 1, 1, 1, 1)
	local hstack = GAU(2*nOutputPlane)(nn.CAddTable()({hconv_out, vconv_to_hconv})) -- - nn.CAddTable() - GAU(2*nOutputPlane)
	-- hstack output is gated elementwise addition of vconv output (convolved w/ 1x1 filter) and hconv output

	-- Add residual on horizontal stack
	local hstack_out = nn.CAddTable()({hstack_in, hstack}) -- - nn.CAddTable()

	return nn.gModule({vstack_in, hstack_in}, {vstack_out, hstack_out})
end


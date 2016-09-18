-- A convolution layer with a gated activation unit as described in the Conditional PixelCNN paper.
-- This convolution layer includes a vertical/horizontal stack, and is intended for use in image generation.

require('nn')


local GatedPixelConvolution, = torch.class('nn.GatedPixelConvolution', 'nn.Module')

function GAU(planes, left, right, batchMode)
	batchMode = opt.batchMode or false
	left_type = opt.left or nn.Tanh
	right_type = opt.right or nn.Sigmoid

	local gate
	gate = nn.Sequential()

	local left = nn.Sequential()
	left:add(nn.NarrowTable(1, planes/2))
	left:add(left_type())
	left:add(nn.JoinTable())

	local right = nn.Sequential()
	right:add(nn.Narrow(planes/2, planes))
	right:add(right_type())
	right:add(nn.JoinTable())

	local concat = nn.ConcatTable()
	concat:add(left)
	concat:add(right)

	local split_dimension = 4
	if batchMode then
		split_dimension = 5
	end

	gate:add(nn.SplitTable(4, 4))
	gate:add(concat)
	gate:add(nn.CMulTable())

	return gate
end

function GatedPixelConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, batchMode)
	batchMode = batchMode or false

	local vertical_conv = nn.SpatialConvolution(ninputPlane, 2*nOutputPlane, kW, kH, dW, dH, padW, padH)
	local horiz_conv = nn.SpatialConvolution(nInputPlane, 2*nOutputPlane, 1, kH, dW, dH, padW, padH)
	-- output of vertical_conv is w x h x nOutputPlane

	local layer = nn.Concat()

	local vertical_s


end
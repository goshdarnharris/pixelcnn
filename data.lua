
require('csvigo')
require('image')

-- Partition data URLs into training and test sets
-- From each of these we can sample a random image with a random mask

data = {}

function download(url, basename, cachepath)
	local ret
	local cache = cachepath or "cache"
	local cachedir = string.format("%s/%d/", cache, basename)

	if path.exists(string.format("%s/frame-1.t7", cachedir)) then
		-- print(string.format("already downloaded gif %d", basename))
		return
	else
		-- print(string.format("downloading gif %d", basename))
		os.execute(string.format("mkdir -p %s", cachedir))
		success, _, ret = os.execute(string.format("wget %s -O %s/image.gif -q", url, cachedir))
		if ret ~= 0 then
			error("could not download image from " .. url)
		end
		success, _, ret = os.execute(string.format("convert -coalesce %s/image.gif %s/frame.jpg", cachedir, cachedir))
		if ret ~= 0 then
			error("could not convert gif to frames")
		end

		local filename = string.format("%s/frame-1.jpg", cachedir)
		local frame = image.load(filename, 3, 'double')
		frame = image.rgb2hsv(frame)
		torch.save(string.format("%s/frame-1.t7", cachedir), frame)
	end
end

function loadimage(t, key)
	local filename = string.format("%s/%d/frame-1.t7", t.cache, key + t.offset)
	local frame = torch.load(filename)
	frame = image.scale(frame, t.width, t.height)

	return frame
end


local data_metatable = {
	__index = loadimage
}

function data.load(path, width, height, size, split, cache)
	split = split or .9

	local data = csvigo.load({path = path, separator = "\t", mode = "large", verbose=false})

	print("checking data cache...")
	for i=1,size do
		io.write(string.format("\rchecking gif %d...", i))
		io.flush()
		download(data[i][1], i, cache)
	end

	local trainsize = math.floor(size * split)
	local testsize = math.ceil(size * (1 - split))

	local train = {
		size = trainsize,
		data = {
			cache = cache,
			width = width,
			height = height,
			offset = 0,
			size = trainsize
		}
	}

	local test = {
		size = testsize,
		data = {
			cache = cache,
			width = width,
			height = height,
			offset = trainsize,
			size = testsize
		}
	}

	setmetatable(train.data, data_metatable)
	setmetatable(test.data, data_metatable)

	print("\ndata loaded.")

	return train, test
end

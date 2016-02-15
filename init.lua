require "torch"
require "cutorch"
require 'ztorch'
-- include('THZCi.lua')
-- local THZCudaTensor_polar = C['THZCudaTensor_polar']
local zcutorch = paths.require("libzcutorch")

local ffi = require 'ffi'
local argcheck = require 'argcheck'

torch.ZCudaStorage.__tostring__ = torch.ZFloatStorage.__tostring__
torch.ZCudaTensor.__tostring__ = torch.ZFloatTensor.__tostring__


include('Tensor.lua')
include('test.lua')

local unpack = unpack or table.unpack

function zcutorch.withDevice(newDeviceID, closure)
    local curDeviceID = cutorch.getDevice()
    zcutorch.setDevice(newDeviceID)
    local vals = {pcall(closure)}
    zcutorch.setDevice(curDeviceID)
    if vals[1] then
        return unpack(vals, 2)
    end
    error(unpack(vals, 2))
end

-- Creates a FloatTensor using the CudaHostAllocator.
-- Accepts either a LongStorage or a sequence of numbers.
function zcutorch.createZCudaHostTensor(...)
   local size
   if not ... then
      size = torch.LongTensor{0}
   elseif torch.isStorage(...) then
      size = torch.LongTensor(...)
   else
      size = torch.LongTensor{...}
   end

   local storage = torch.ZFloatStorage(zcutorch.CudaHostAllocator, size:prod())
   return torch.ZFloatTensor(storage, 1, size:storage())
end

zcutorch.setHeapTracking(true)

-- zcutorch.polar = argcheck{
--    nonamed=true,
--    {name="src1", type=ctypename},
--    {name="src2", type=ctypename},
--    call = function(src1, src2)
--       ret = torch.ZCudaTensor(src1:size())
--       THZCudaTensor_polar(cutorch._state,ret, src1, src2)
--       return ret
--    end
-- }
--
-- torch.polar = argcheck{
--    nonamed=true,
--    {name="src1", type=ctypename},
--    {name="src2", type=ctypename},
--    call = function(src1, src2)
--       ret = torch.ZCudaTensor(src1:size())
--       THZCudaTensor_polar(cutorch._state,ret, src1, src2)
--       return ret
--    end
-- }

return zcutorch

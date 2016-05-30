include('THZCi.lua')
local argcheck = require 'argcheck'
local ffi=require 'ffi'
-- local torch = require 'torch'
require 'pprint'
local ztorch = require 'ztorch'

local THZCudaTensor_abs = C['THZCudaTensor_abs']
local THZCudaTensor_arg = C['THZCudaTensor_arg']
local THZCudaTensor_norm = C['THZCudaTensor_norm']

local THZCudaTensor_real = C['THZCudaTensor_real']
local THZCudaTensor_imag = C['THZCudaTensor_imag']
local THZCudaTensor_sign = C['THZCudaTensor_sign']

local THZCudaTensor_zabs = C['THZCudaTensor_zabs']
local THZCudaTensor_zarg = C['THZCudaTensor_zarg']
local THZCudaTensor_znorm = C['THZCudaTensor_znorm']

local THZCudaTensor_zreal = C['THZCudaTensor_zreal']
local THZCudaTensor_zimag = C['THZCudaTensor_zimag']

local THZCudaTensor_normall = C['THZCudaTensor_normall']
local THZCudaTensor_normDim = C['THZCudaTensor_normDim']


local THZCudaTensor_cim = C['THZCudaTensor_cim']
local THZCudaTensor_cre = C['THZCudaTensor_cre']

local THZCudaTensor_copyIm = C['THZCudaTensor_copyIm']
local THZCudaTensor_copyRe = C['THZCudaTensor_copyRe']

local THZCudaTensor_fillim = C['THZCudaTensor_fillim']
local THZCudaTensor_fillre = C['THZCudaTensor_fillre']

local THZCudaTensor_polar = C['THZCudaTensor_polar']


local THZCudaTensor_copy = C['THZCudaTensor_copy']
local THZCudaTensor_copyByte = C['THZCudaTensor_copyByte']
local THZCudaTensor_copyChar = C['THZCudaTensor_copyChar']
local THZCudaTensor_copyShort = C['THZCudaTensor_copyShort']
local THZCudaTensor_copyInt = C['THZCudaTensor_copyInt']
local THZCudaTensor_copyLong = C['THZCudaTensor_copyLong']
local THZCudaTensor_copyFloat = C['THZCudaTensor_copyFloat']
local THZCudaTensor_copyDouble = C['THZCudaTensor_copyDouble']
local THZCudaTensor_copyZFloat = C['THZCudaTensor_copyZFloat']
local THZFloatTensor_copyZCuda = C['THZFloatTensor_copyZCuda']
local THZCudaTensor_copyZCuda = C['THZCudaTensor_copyZCuda']
local THZCudaTensor_copyAsyncZFloat = C['THZCudaTensor_copyAsyncZFloat']
local THZFloatTensor_copyAsyncZCuda = C['THZFloatTensor_copyAsyncZCuda']

local THZCudaTensor_fft = C['THZCudaTensor_fft']
local THZCudaTensor_fftBatched = C['THZCudaTensor_fftBatched']
local THZCudaTensor_ifft = C['THZCudaTensor_ifft']
local THZCudaTensor_ifftBatched = C['THZCudaTensor_ifftBatched']
local THZCudaTensor_ifftU = C['THZCudaTensor_ifftU']
local THZCudaTensor_ifftBatchedU = C['THZCudaTensor_ifftBatchedU']

local THZCudaTensor_add = C['THZCudaTensor_add']
local THZCudaTensor_mul = C['THZCudaTensor_mul']
local THZCudaTensor_div = C['THZCudaTensor_div']
local THZCudaTensor_addcmul = C['THZCudaTensor_addcmul']
local THZCudaTensor_addcdiv = C['THZCudaTensor_addcdiv']
local THZCudaTensor_cmul = C['THZCudaTensor_cmul']
local THZCudaTensor_cmulZR = C['THZCudaTensor_cmulZR']
local THZCudaTensor_cadd = C['THZCudaTensor_cadd']
local THZCudaTensor_cpow = C['THZCudaTensor_cpow']
local THZCudaTensor_cdiv = C['THZCudaTensor_cdiv']
local THZCudaTensor_cdivZR = C['THZCudaTensor_cdivZR']

local THZCudaTensor_narrow = C['THZCudaTensor_narrow']
local THZCudaTensor_select = C['THZCudaTensor_select']

local THZCudaTensor_newNarrow = C['THZCudaTensor_newNarrow']
local THZCudaTensor_newSelect = C['THZCudaTensor_newSelect']

local THZCudaTensor_resize = C['THZCudaTensor_resize']
local THZCudaTensor_resizeAs = C['THZCudaTensor_resizeAs']
local THZCudaTensor_resize4d = C['THZCudaTensor_resize4d']
local THZCudaTensor_dot = C['THZCudaTensor_dot']
local THZCudaTensor_maskedFill = C['THZCudaTensor_maskedFill']

function torch.ZCudaTensor.apply(self, func)
   local x = torch.ZFloatTensor(self:size()):copy(self)
   x:apply(func)
   self:copy(x)
   return self
end

local function Tensor__type(self,type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:nElement() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end
local function Tensor__typeAs(self,tensor)
   return self:type(tensor:type())
end

local function Tensor__zcuda(self)
  local zfloat = self:type('torch.ZFloatTensor')
  return Tensor__type(zfloat,"torch.ZCudaTensor")
end

local function ZFTensor__zcuda(self)
  return torch.ZCudaTensor(self:size()):copy(self)
end

local function Tensor__zcuda_device(self)

  -- TODO
end

local function Tensor__double(self)
   return self:type('torch.DoubleTensor')
end

local function Tensor__zfloat(self)
   return self:type('torch.ZFloatTensor')
end

local function Tensor__byte(self)
   return self:type('torch.ByteTensor')
end

local function Tensor__char(self)
   return self:type('torch.CharTensor')
end

local function Tensor__int(self)
   return self:type('torch.IntTensor')
end

local function Tensor__short(self)
   return self:type('torch.ShortTensor')
end

local function Tensor__long(self)
   return self:type('torch.LongTensor')
end

local ZTensor = {}

local typename = "torch.ZCudaTensor"
local ctypename = "torch.CudaTensor"
local storageType = 'torch.ZCudaStorage'

local THZCudaTensor_new = C['THZCudaTensor_new']
local THZCudaTensor_newWithTensor = C['THZCudaTensor_newWithTensor']
local THZCudaTensor_newWithStorage = C['THZCudaTensor_newWithStorage']
local THZCudaTensor_newWithSize4d = C['THZCudaTensor_newWithSize4d']
local THZCudaTensor_newClone = C['THZCudaTensor_newClone']
local THZCudaTensor_free = C['THZCudaTensor_free']
local THZCudaTensor_newWithSize = C['THZCudaTensor_newWithSize']

local function free(t)
  THZCudaTensor_free(cutorch._state,t)
end

local function wrapcdata(cdata)
  -- print('wrapcdata')
  return torch.ZCudaTensor().wrapcdata(cdata)
end

ZTensor.__new = argcheck{
   nonamed=true,
   call =
      function()
         local self = wrapcdata(THZCudaTensor_new(cutorch._state))
        --  ffi.gc(self, THZCudaTensor_free)
         return self
      end
}

ZTensor.__new = argcheck{
   {name='storage', type=storageType},
   {name='storageOffset', type='number', default=1},
   {name='size', type='table', opt=true},
   {name='stride', type='table', opt=true},
   nonamed=true,
   overload = ZTensor.__new,
   call =
      function(storage, storageOffset, size, stride)
         if size then
            size = torch.LongStorage(size):cdata()
         end
         if stride then
            stride = torch.LongStorage(stride):cdata()
         end
         local self = wrapcdata(THZCudaTensor_newWithStorage(cutorch._state,storage:cdata(), storageOffset-1, size, stride))
        --  ffi.gc(self, free)
         return self
      end
}

ZTensor.__new = argcheck{
   {name='dim1', type='number'},
   {name='dim2', type='number', default=0},
   {name='dim3', type='number', default=0},
   {name='dim4', type='number', default=0},
   nonamed=true,
   overload = ZTensor.__new,
   call =
      function(dim1, dim2, dim3, dim4)
         local self = wrapcdata(THZCudaTensor_newWithSize4d(cutorch._state,dim1, dim2, dim3, dim4))
        --  print(type(self))
        --  print(self)
        --  ffi.gc(self, free)
         return self
      end
}
ZTensor.__new = argcheck{
   {name='dimt', type='table'},
   nonamed=true,
   overload = ZTensor.__new,
   call =
      function(dimt)
         local dim1 = dimt[1] or 0
         local dim2 = dimt[2] or 0
         local dim3 = dimt[3] or 0
         local dim4 = dimt[4] or 0
        --  pprint(dimt)
         local self = wrapcdata(THZCudaTensor_newWithSize4d(cutorch._state,dim1, dim2, dim3, dim4))
        --  self = torch.ZCudaTensor(self)
        --  print(type(self))
        --  print(self)
        --  ffi.gc(self, free)
         return self
      end
}

ZTensor.__new = argcheck{
   {name='size', type='torch.LongStorage'},
   nonamed=true,
   overload = ZTensor.__new,
   call =
      function(size)
         local self = wrapcdata(THZCudaTensor_newWithSize(cutorch._state,size:cdata(), nil))
        --  ffi.gc(self, free)
         return self
      end
}

ZTensor.__new = argcheck{
   {name='tensor', type=typename},
   nonamed=true,
   overload = ZTensor.__new,
   call =
      function(tensor)
        --  print('new with tensor')
         local self = wrapcdata(THZCudaTensor_newWithTensor(cutorch._state,tensor:cdata()))
        --  ffi.gc(self, free)
         return self
      end
}

ZTensor.newClone = argcheck{
   {name='tensor', type=typename},
   nonamed=true,
   call =
      function(tensor)
         local self = wrapcdata(THZCudaTensor_newClone(cutorch._state,tensor:cdata()))
        --  ffi.gc(self, free)
         return self
      end
}

ZTensor.__call = ZTensor.__new
ZTensor.new = ZTensor.__new

ZTensor.copyIm = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=ctypename},
   call = function(dst, src)
      THZCudaTensor_copyIm(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}

ZTensor.copyRe = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=ctypename},
   call = function(dst, src)
      THZCudaTensor_copyRe(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}

ZTensor.fillIm = argcheck{
   nonamed=true,
   {name="src", type=typename},
   {name="v", type="number"},
   call = function( src, v)
      THZCudaTensor_fillim(cutorch._state, src:cdata(),v)
      return src
   end
}

ZTensor.fillRe = argcheck{
   nonamed=true,
   {name="src", type=typename},
   {name="v", type="number"},
   call = function( src, v)
      THZCudaTensor_fillre(cutorch._state, src:cdata(),v)
      return src
   end
}

ZTensor.abs = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = torch.CudaTensor()
      THZCudaTensor_abs(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.abs = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename},
   overload=ZTensor.abs,
   call = function(dst, src)
      THZCudaTensor_zabs(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.abs = argcheck{
   nonamed=true,
   {name="dst", type=ctypename},
   {name="src", type=typename},
   overload=ZTensor.abs,
   call = function(dst, src)
      THZCudaTensor_abs(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}

ZTensor.arg = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = torch.CudaTensor()
      THZCudaTensor_arg(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}

ZTensor.arg = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename},
   overload=ZTensor.arg,
   call = function(dst, src)
      THZCudaTensor_zarg(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.arg = argcheck{
   nonamed=true,
   {name="dst", type=ctypename},
   {name="src", type=typename},
   overload=ZTensor.arg,
   call = function(dst, src)
      THZCudaTensor_arg(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.re = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = torch.CudaTensor()
      THZCudaTensor_real(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.re = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename},
   overload=ZTensor.re,
   call = function(dst, src)
      THZCudaTensor_zreal(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.re = argcheck{
   nonamed=true,
   {name="dst", type=ctypename},
   {name="src", type=typename},
   overload=ZTensor.re,
   call = function(dst, src)
      THZCudaTensor_real(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.sign = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = torch.CudaTensor()
      THZCudaTensor_sign(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}

ZTensor.im = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = torch.CudaTensor()
      THZCudaTensor_imag(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.im = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename},
   overload=ZTensor.im,
   call = function(dst, src)
      THZCudaTensor_zimag(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.im = argcheck{
   nonamed=true,
   {name="dst", type=ctypename},
   {name="src", type=typename},
   overload=ZTensor.im,
   call = function(dst, src)
      THZCudaTensor_imag(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.norm = argcheck{
   nonamed=true,
   {name="dst", type=ctypename},
   {name="src", type=typename},
   overload=ZTensor.norm,
   call = function(dst, src)
      THZCudaTensor_norm(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}
ZTensor.norm = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename, opt=true},
   overload=ZTensor.norm,
   call = function(dst, src)
      src = src or dst
      THZCudaTensor_znorm(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}

ZTensor.normall = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="value", type="number"},
   call = function(self, value)
      return THZCudaTensor_normall(cutorch._state,self:cdata(), value)
   end
}

ZTensor.normDim = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src", type=typename},
   {name="value", type="number"},
   {name="dimension", type="number"},
   call = function(self, src, value, dimension)
      THZCudaTensor_normDim(cutorch._state,self:cdata(), src:cdata(), value, dimension)
      return self
   end
}

ZTensor.polar = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src1", type=ctypename},
   {name="src2", type=ctypename},
   call = function(self, src1, src2)
      THZCudaTensor_polar(cutorch._state,self:cdata(), src1:cdata(), src2:cdata())
      return self
   end
}
ZTensor.polar = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src1", type="number"},
   {name="src2", type=ctypename},
   overload=ZTensor.polar,
   call = function(self, src1, src2)
      local s1 = src2:clone():fill(src1)
      THZCudaTensor_polar(cutorch._state,self:cdata(), s1:cdata(), src2:cdata())
      return self
   end
}
ZTensor.polar = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src1", type=ctypename},
   {name="src2", type="number"},
   overload=ZTensor.polar,
   call = function(self, src1, src2)
      local s2 = src1:clone():fill(src2)
      THZCudaTensor_polar(cutorch._state,self:cdata(), src1:cdata(), s2:cdata())
      return self
   end
}

ZTensor.cim = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src2", type=ctypename},
   call = function(self, src2)
      THZCudaTensor_cim(cutorch._state,self:cdata(), self:cdata(), src2:cdata())
      return self
   end
}
ZTensor.cim = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src1", type=typename},
   {name="src2", type=ctypename},
   overload=ZTensor.cim,
   call = function(self, src1, src2)
      THZCudaTensor_cim(cutorch._state,self:cdata(), src1:cdata(), src2:cdata())
      return self
   end
}

ZTensor.cre = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src2", type=ctypename},
   call = function(self, src2)
      THZCudaTensor_cre(cutorch._state,self:cdata(), self:cdata(), src2:cdata())
      return self
   end
}

ZTensor.cre = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src1", type=typename},
   {name="src2", type=ctypename},
   overload=ZTensor.cre,
   call = function(self, src1, src2)
      THZCudaTensor_cre(cutorch._state,self:cdata(), src1:cdata(), src2:cdata())
      return self
   end
}
-- local THZCudaTensor_fft = C['THZCudaTensor_fft']
-- local THZCudaTensor_fftBatched = C['THZCudaTensor_fftBatched']
-- local THZCudaTensor_ifft = C['THZCudaTensor_ifft']
-- local THZCudaTensor_ifftBatched = C['THZCudaTensor_ifftBatched']
-- local THZCudaTensor_ifftU = C['THZCudaTensor_ifftU']
-- local THZCudaTensor_ifftBatchedU = C['THZCudaTensor_ifftBatchedU']
-- local THZCudaTensor_fftShiftedInplace = C['THZCudaTensor_fftShiftedInplace']
-- local THZCudaTensor_fftShifted = C['THZCudaTensor_fftShifted']
-- local THZCudaTensor_fftShiftInplace = C['THZCudaTensor_fftShiftInplace']
-- local THZCudaTensor_fftShift = C['THZCudaTensor_fftShift']
-- local THZCudaTensor_ifftShiftInplace = C['THZCudaTensor_ifftShiftInplace']
-- local THZCudaTensor_ifftShift = C['THZCudaTensor_ifftShift']
ZTensor.dot = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src1", type=typename, opt=true},
   call =
      function(dst, src1)
         src1 = src1 or dst
         local result = THZCudaTensor_dot(cutorch._state,src1:cdata(), dst:cdata())
        --  print('in dot ',result)
         return result
      end
}
ZTensor.fft = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src1", type=typename, opt=true},
   call =
      function(dst, src1)
        --  print('in fft')
        --  pprint(dst)
        --  pprint(src1)
         src1 = src1 or dst
         THZCudaTensor_fft(cutorch._state,src1:cdata(), dst:cdata())
        --  print('end fft')
         return dst
      end
}
ZTensor.fftshift = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src1", type=typename, opt=true},
   call =
      function(dst, src1)
        if src1 then
          dst:copy(src1)
        end
        local ndim = dst:dim()

        local t = torch.getdefaulttensortype()
        torch.setdefaulttensortype('torch.FloatTensor')
        local axes = torch.linspace(1,ndim,ndim)
        torch.setdefaulttensortype(t)
        for _, k in pairs(axes:totable()) do
          local n = dst:size(k)
          local p2 = math.floor((n+1)/2)

          local half1 = {p2+1,n}
          local half2 = {1,p2}
          -- pprint(half1)
          -- pprint(half2)

          local indextable = {{},{}}
      --    pprint(indextable)
          for i=1,ndim do
      --      print(k .. '_' .. i)
            if i ~= k then
              indextable[1][i] = {}
              indextable[2][i] = {}

      --        pprint(indextable)
            else
              indextable[1][i] = half1
              indextable[2][i] = half2
      --        pprint(indextable)
            end
          end
          -- pprint(indextable[1])
          -- pprint(indextable[2])
          local tmp = dst[indextable[1]]:clone()
      --    pprint(tmp)
      --    pprint(dst)
          dst[indextable[1]]:copy(dst[indextable[2]])
          dst[indextable[2]]:copy(tmp)
        end
        return dst
      end
}
ZTensor.ifftshift = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src1", type=typename, opt=true},
   call =
      function(dst, src1)
        if src1 then
          dst:copy(src1)
        end
        local ndim = dst:dim()
        local t = torch.getdefaulttensortype()
        torch.setdefaulttensortype('torch.FloatTensor')
        local axes = torch.linspace(1,ndim,ndim)
        torch.setdefaulttensortype(t)
        for _, k in pairs(axes:totable()) do
          local n = dst:size(k)
          local p2 = math.floor(n-(n+1)/2)

          local half1 = {p2+1,n}
          local half2 = {1,p2}
      --    pprint(half1)
      --    pprint(half2)

          local indextable = {{},{}}
      --    pprint(indextable)
          for i=1,ndim do
      --      print(k .. '_' .. i)
            if i ~= k then
              indextable[1][i] = {}
              indextable[2][i] = {}

      --        pprint(indextable)
            else
              indextable[1][i] = half1
              indextable[2][i] = half2
      --        pprint(indextable)
            end
          end
      --    pprint(indextable[1])
      --    pprint(indextable[2])
          local tmp = dst[indextable[1]]:clone()
      --    pprint(tmp)
      --    pprint(dst)
          dst[indextable[1]]:copy(dst[indextable[2]])
          dst[indextable[2]]:copy(tmp)
        end
        return dst
      end
}
ZTensor.fftBatched = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   call =
      function(dst, src1)
         dst = dst or src1
         THZCudaTensor_fftBatched(cutorch._state,src1:cdata(), dst:cdata())
         return dst
      end
}

ZTensor.ifft = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   call =
      function(dst, src1)
         dst = dst or src1
         THZCudaTensor_ifftU(cutorch._state,src1:cdata(), dst:cdata())
         THZCudaTensor_mul(cutorch._state,dst:cdata(), src1:cdata(),(1/dst:nElement(0) + 0i))
         return dst
      end
}

ZTensor.ifftBatched = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   call =
      function(dst, src1)
         dst = dst or src1
         THZCudaTensor_ifftBatchedU(cutorch._state,src1:cdata(), dst:cdata())
         THZCudaTensor_mul(cutorch._state,dst:cdata(), src1:cdata(),(1/dst:nElement(0) + 0i))
         return dst
      end
}

ZTensor.ifftU = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   call =
      function(dst, src1)
         dst = dst or src1
         THZCudaTensor_ifftU(cutorch._state,src1:cdata(), dst:cdata())
         return dst
      end
}

ZTensor.ifftBatchedU = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   call =
      function(dst, src1)
         dst = dst or src1
         THZCudaTensor_ifftBatchedU(cutorch._state,src1:cdata(), dst:cdata())
         return dst
      end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type=typename},
   call =
      function(dst, src)
        -- print('copy THZCudaTensor_copy')
         THZCudaTensor_copy(cutorch._state,dst:cdata(), src:cdata())
         return dst
      end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type='torch.ByteTensor'},
   overload=ZTensor.copy,
   call =
      function(dst, src)
         THZCudaTensor_copyByte(cutorch._state,dst:cdata(), src:cdata())
         return dst
      end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type='torch.CharTensor'},
   overload=ZTensor.copy,
   call =
      function(dst, src)
         THZCudaTensor_copyChar(cutorch._state,dst:cdata(), src:cdata())
         return dst
      end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type='torch.ShortTensor'},
   overload=ZTensor.copy,
   call =
      function(dst, src)
         THZCudaTensor_copyShort(cutorch._state,dst:cdata(), src:cdata())
         return dst
      end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type='torch.IntTensor'},
   overload=ZTensor.copy,
   call =
      function(dst, src)
         THZCudaTensor_copyInt(cutorch._state,dst:cdata(), src:cdata())
         return dst
      end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type='torch.LongTensor'},
   overload=ZTensor.copy,
   call =
      function(dst, src)
        -- print('copy 5')
         THZCudaTensor_copyLong(cutorch._state,dst:cdata(), src:cdata())
         return dst
      end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type='torch.FloatTensor'},
   overload=ZTensor.copy,
   call =
      function(dst, src)
        -- print('copy 4')
         THZCudaTensor_copyFloat(cutorch._state,dst:cdata(), src:cdata())
         return dst
      end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type='torch.ZFloatTensor'},
   overload=ZTensor.copy,
   call =
      function(dst, src)
        -- print('copy zfloat')
         THZCudaTensor_copyZFloat(cutorch._state,dst:cdata(), src)
         return dst
      end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type='torch.DoubleTensor'},
   overload=ZTensor.copy,
   call =
      function(dst, src)
        -- print('copy 3')
         THZCudaTensor_copyDouble(dst:cdata(), src:cdata())
         return dst
      end
}
ZTensor.mul = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src", type=typename},
   {name="value", type="number"},
   call =
      function(dst, src, value)
         dst = dst or src
         THZCudaTensor_mul(cutorch._state, dst:cdata(), src:cdata(), value)
         return dst
      end
}
ZTensor.mul = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src", type=typename},
   {name="value", type="cdata", check=ztorch.isComplex},
   overload=ZTensor.mul,
   call =
      function(dst, src, value)
         dst = dst or src
         THZCudaTensor_mul(cutorch._state, dst:cdata(), src:cdata(), value)
         return dst
      end
}
ZTensor.add = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   {name="v", type='number'},
   call =
      function(dst, src1, v)
         dst = dst or src1
         THZCudaTensor_add(cutorch._state,dst:cdata(), src1:cdata(),v)
         return dst
      end
}
ZTensor.add = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   {name="v", type="cdata", check=ztorch.isComplex},
   call =
      function(dst, src1, v)
         dst = dst or src1
         THZCudaTensor_add(cutorch._state,dst:cdata(), src1:cdata(),v)
         return dst
      end
}
ZTensor.maskedFill = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="mask", type=ctypename},
   {name="v", type="cdata", check=ztorch.isComplex},
   call =
      function(dst, mask, v)
         THZCudaTensor_maskedFill(cutorch._state,dst:cdata(), mask:cdata(),v)
         return dst
      end
}
ZTensor.cmul = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   {name="src2", type=typename},
   call =
      function(dst, src1, src2)
         dst = dst or src1
         THZCudaTensor_cmul(cutorch._state,dst:cdata(), src1:cdata(), src2:cdata())
         return dst
      end
}
ZTensor.cmul = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   {name="src2", type=ctypename},
   overload=ZTensor.cmul,
   call =
      function(dst, src1, src2)
         dst = dst or src1
         THZCudaTensor_cmulZR(cutorch._state,dst:cdata(), src1:cdata(), src2:cdata())
         return dst
      end
}
ZTensor.cdiv = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   {name="src2", type=typename},
   call =
      function(dst, src1, src2)
         dst = dst or src1
         THZCudaTensor_cdiv(cutorch._state,dst:cdata(), src1:cdata(), src2:cdata())
         return dst
      end
}
ZTensor.cdiv = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   {name="src2", type=ctypename},
   overload=ZTensor.cdiv,
   call =
      function(dst, src1, src2)
         dst = dst or src1
         THZCudaTensor_cdivZR(cutorch._state,dst:cdata(), src1:cdata(), src2:cdata())
         return dst
      end
}
ZTensor.add = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   {name="v", type='number', default=1+0i},
   {name="src2", type=typename},
   overload=ZTensor.add,
   call =
      function(dst, src1, v, src2)
         dst = dst or src1
         THZCudaTensor_cadd(cutorch._state,dst:cdata(), src1:cdata(),v, src2:cdata())
         return dst
      end
}

ZTensor.addcmul = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   {name="v", type='number', default=1+0i},
   {name="src2", type=typename},
   {name="src3", type=typename},
   call =
      function(dst, src1, v, src2, src3)
         dst = dst or src1
         THZCudaTensor_addcmul(cutorch._state,dst:cdata(), src1:cdata(),v, src2:cdata(), src3:cdata())
         return dst
      end
}
ZTensor.addcdiv = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   {name="v", type='number'},
   {name="src2", type=typename},
   {name="src3", type=typename},
   call =
      function(dst, src1, v, src2, src3)
         dst = dst or src1
         THZCudaTensor_addcdiv(cutorch._state,dst:cdata(), src1:cdata(),v, src2:cdata(), src3:cdata())
         return dst
      end
}
-- THCState *state, THZCudaTensor *self,
--                                    THZCudaTensor *src, int dimension_,
--                                    long sliceIndex_


ZTensor.resize = argcheck{
   {name='self', type=typename},
   {name='size', type='table'},
   {name='stride', type='table', opt=true},
   nonamed=true,
   call =
      function(self, size, stride)
         local dim = #size
         assert(not stride or (#stride == dim), 'inconsistent size/stride sizes')
         size = torch.LongStorage(size)
         local stridecdata
         if stride then
            stride = torch.LongStorage(stride)
            stridecdata = stride:cdata()
         end
         THZCudaTensor_resize(cutorch._state,self:cdata(), size:cdata(), stridecdata)
         return self
      end
}

ZTensor.resize = argcheck{
   {name='self', type=typename},
   {name='dim1', type='number'},
   {name='dim2', type='number', default=0},
   {name='dim3', type='number', default=0},
   {name='dim4', type='number', default=0},
   nonamed=true,
   overload = ZTensor.resize,
   call =
      function(self, dim1, dim2, dim3, dim4)
         THZCudaTensor_resize4d(cutorch._state,self:cdata(), dim1, dim2, dim3, dim4)
         return self
      end
}

ZTensor.resize = argcheck{
   {name='self', type=typename},
   {name='size', type='torch.LongStorage'},
   {name='stride', type='torch.LongStorage', opt=true},
   nonamed=true,
   overload = ZTensor.resize,
   call =
      function(self, size, stride)
         if stride then stride = stride:cdata() end
         THZCudaTensor_resize(cutorch._state,self:cdata(), size:cdata(), stride)
         return self
      end
}

ZTensor.resizeAs = argcheck{
   {name='self', type=typename},
   {name='src', type=typename},
   nonamed=true,
   call =
      function(self, src)
         THZCudaTensor_resizeAs(cutorch._state,self:cdata(), src:cdata())
         return self
      end
}

ZTensor.resizeAs = argcheck{
   {name='self', type=typename},
   {name='src', type=ctypename},
   overload = ZTensor.resizeAs,
   nonamed=true,
   call =
      function(self, src)
         THZCudaTensor_resize(cutorch._state,self:cdata(), src:size():cdata(), src:stride():cdata())
         return self
      end
}

ZTensor.select = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src", type=typename, opt=true},
   {name="dim", type='number'},
   {name="index", type='number'},
   call =
      function(self, src, dim, index)
        src = src or self
        if index < 0 then
          index = src:size(dim) + index + 1
        else
          index = index - 1
        end
        -- pprint(src:size())
        -- print('dim index ' .. dim .. ' ' .. index)
        assert(dim <= src:dim(), 'dim must be within source dimensions. dim = ' .. dim)
        assert(index <= src:size(dim), 'index must be within source dimensions. ind = ' .. index)
        -- THZCudaTensor_select(cutorch._state,self:cdata(),src:cdata(),dim,index)
        -- return self
        local ret = wrapcdata(THZCudaTensor_newSelect(cutorch._state,src:cdata(),dim-1,index))
        return ret
      end
}

ZTensor.newSelect = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src", type=typename, opt=true},
   {name="dim", type='number'},
   {name="index", type='number'},
   call =
      function(self, src, dim, index)
        src = src or self
        if index < 0 then
          index = src:size(dim) + index + 1
        else
          index = index - 1
        end
        assert(dim <= src:dim(), 'dim must be within source dimensions. dim = ' .. dim)
        assert(index <= src:size(dim), 'index must be within source dimensions. ind = ' .. index)
        local ret = wrapcdata(THZCudaTensor_newSelect(cutorch._state,src:cdata(),dim-1,index))
        return ret
      end
}
-- THCState *state, THZCudaTensor *self,
--                                    THZCudaTensor *src, int dimension_,
--                                    long firstIndex_, long size_
ZTensor.narrow = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src", type=typename, opt=true},
   {name="dim", type='number'},
   {name="start", type='number'},
   {name="size", type='number'},
   call =
      function(self, src, dim, start, size)
          src = src or self

          assert(dim >= 0 and dim <= src:dim(), 'dim must be within source dimensions')
          assert(start <= src:size(dim) and start >= 0, 'start must be within source dimensions. start = ' .. start)
          assert(size <= src:size(dim) and size >= 0, 'size must be within source dimensions. size = ' .. size)
          assert(start-1 + size <= src:size(dim), 'start+size must be within source dimensions. s+s-1 = ' .. (start-1+size) .. ' size = ' .. src:size(dim))
          THZCudaTensor_narrow(cutorch._state,self:cdata(),src:cdata(),dim-1,start-1,size)
          return self
      end
}
ZTensor.newNarrow = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src", type=typename, opt=true},
   {name="dim", type='number'},
   {name="start", type='number'},
   {name="size", type='number'},
   call =
      function(self, src, dim, start, size)
          src = src or self

          assert(dim >= 0 and dim <= src:dim(), 'dim must be within source dimensions')
          assert(start <= src:size(dim) and start >= 0, 'start must be within source dimensions. start = ' .. start)
          assert(size <= src:size(dim) and size >= 0, 'size must be within source dimensions. size = ' .. size)
          assert(start-1 + size <= src:size(dim), 'start+size-1 must be within source dimensions. s+s = ' .. (start+size-1) .. ' size = ' .. src:size(dim))
          local ret = wrapcdata(THZCudaTensor_newNarrow(cutorch._state,self:cdata(),src:cdata(),dim-1,start-1,size))
          return ret
      end
}

ZTensor.__index = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="ind", type='table'},
   call =
      function(self, ind)
        local cdim = 1
        -- print('in __index 2')
        local ret = ZTensor.new(self)
        for _, v in ipairs(ind) do
          if type(v) == 'number' then
            -- somethig
            ret = ret:select(cdim,v)
          elseif type(v) == 'table' then
            local start = (v[1] or 1)
            local ends = (v[2] or self:size(cdim))
            if ends < 0 then
              ends = self:size(cdim) + ends + 1
            end
            local size = ends - start + 1
            ret = ret:narrow(cdim,start,size)
            cdim = cdim + 1
          end
        end
        return ret
      end
}
ZTensor.__index = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="ind", type='number'},
   overload = ZTensor.__index,
   call =
      function(self, ind)
        -- print('in __index 0')
        local ret = ZTensor.new(self)
        -- print('in __index 1')
        -- pprint(ret)
        return ret:select(1,ind)
      end
}
ZTensor.__index = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="s", type='string'},
   overload = ZTensor.__index,
   call =
      function(self, s)
        local t = torch.getmetatable('torch.ZCudaTensor')
        return t[s]
      end
}

local Tensor = {}
Tensor.fftshift = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src1", type='torch.CudaTensor', opt=true},
   call =
      function(dst, src1)
        if src1 then
          dst:copy(src1)
        end
        local ndim = dst:dim()

        local t = torch.getdefaulttensortype()
        torch.setdefaulttensortype('torch.FloatTensor')
        local axes = torch.linspace(1,ndim,ndim)
        -- pprint(axes)
        torch.setdefaulttensortype(t)
        for _, k in pairs(axes:totable()) do
          local n = dst:size(k)
          local p2 = math.floor((n+1)/2)

          local half1 = {p2+1,n}
          local half2 = {1,p2}
        --  pprint(half1)
        --  pprint(half2)

          local indextable = {{},{}}
      --    pprint(indextable)
          for i=1,ndim do
          --  print(k .. '_' .. i)
            if i ~= k then
              indextable[1][i] = {}
              indextable[2][i] = {}
            --  pprint(indextable)
            else
              indextable[1][i] = half1
              indextable[2][i] = half2
            --  pprint(indextable)
            end
          end
          -- pprint(indextable[1])
          -- pprint(indextable[2])
          -- pprint(dst)
          local tmp = dst[indextable[1]]:clone()
        --  pprint(tmp)
        --  pprint(dst)
          dst[indextable[1]]:copy(dst[indextable[2]])
          dst[indextable[2]]:copy(tmp)
        end
        return dst
      end
}
Tensor.ifftshift = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src1", type='torch.CudaTensor', opt=true},
   call =
      function(dst, src1)
        if src1 then
          dst:copy(src1)
        end
        local ndim = dst:dim()
        local t = torch.getdefaulttensortype()
        torch.setdefaulttensortype('torch.FloatTensor')
        local axes = torch.linspace(1,ndim,ndim)
        torch.setdefaulttensortype(t)
        for _, k in pairs(axes:totable()) do
          local n = dst:size(k)
          local p2 = math.floor(n-(n+1)/2)

          local half1 = {p2+1,n}
          local half2 = {1,p2}
      --    pprint(half1)
      --    pprint(half2)

          local indextable = {{},{}}
      --    pprint(indextable)
          for i=1,ndim do
      --      print(k .. '_' .. i)
            if i ~= k then
              indextable[1][i] = {}
              indextable[2][i] = {}
      --        pprint(indextable)
            else
              indextable[1][i] = half1
              indextable[2][i] = half2
      --        pprint(indextable)
            end
          end
      --    pprint(indextable[1])
      --    pprint(indextable[2])
          local tmp = dst[indextable[1]]:clone()
      --    pprint(tmp)
      --    pprint(dst)
          dst[indextable[1]]:copy(dst[indextable[2]])
          dst[indextable[2]]:copy(tmp)
        end
        return dst
      end
}

local zmetatable = torch.getmetatable('torch.ZCudaTensor')
rawset( zmetatable, 'type', Tensor__type)
rawset( zmetatable, 'typeAs', Tensor__typeAs)
rawset( zmetatable, 'zfloat', Tensor__zfloat)

for k,v in pairs(ZTensor) do
  rawset( zmetatable, k, v)
end
-- ffi.metatype('THZCudaTensor', zmetatable)
local zfname = 'torch.ZFloatTensor'
local zfmetatable = torch.getmetatable(zfname)

zfmetatable.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=zfname},
   {name="src", type=typename},
   overload=zfmetatable.copy,
   call =
      function(dst, src)
         THZFloatTensor_copyZCuda(cutorch._state,dst, src:cdata())
         return dst
      end
}

local cmetatable = torch.getmetatable('torch.CudaTensor')
rawset( cmetatable, 'reZ', ZTensor.re)
rawset( cmetatable, 'imZ', ZTensor.im)
rawset( cmetatable, 'argZ', ZTensor.arg)
rawset( cmetatable, 'absZ', ZTensor.abs)
rawset( cmetatable, 'normZ', ZTensor.norm)
rawset( cmetatable, 'fftshift', Tensor.fftshift)
rawset( cmetatable, 'ifftshift', Tensor.ifftshift)


rawset(torch.getmetatable('torch.DoubleTensor'), 'zcuda', Tensor__zcuda)
rawset(torch.getmetatable('torch.FloatTensor'), 'zcuda', Tensor__zcuda)
rawset(torch.getmetatable('torch.ByteTensor'), 'zcuda', Tensor__zcuda)
rawset(torch.getmetatable('torch.CharTensor'), 'zcuda', Tensor__zcuda)
rawset(torch.getmetatable('torch.IntTensor'), 'zcuda', Tensor__zcuda)
rawset(torch.getmetatable('torch.ShortTensor'), 'zcuda', Tensor__zcuda)
rawset(torch.getmetatable('torch.LongTensor'), 'zcuda', Tensor__zcuda)
rawset(torch.getmetatable('torch.ZFloatTensor'), 'zcuda', ZFTensor__zcuda)
rawset(torch.getmetatable('torch.CudaTensor'), 'zcuda', Tensor__zcuda_device)

do
    local metatable = torch.getmetatable('torch.ZCudaTensor')
    for _,func in pairs{'expand', 'expandAs', 'view', 'viewAs', 'repeatTensor',
                        'permute', 'split', 'chunk'} do
        rawset(metatable, func, torch[func])
    end
end

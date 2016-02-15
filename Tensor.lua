include('THZCi.lua')
local argcheck = require 'argcheck'

local THZCudaTensor_abs = C['THZCudaTensor_abs']
local THZCudaTensor_arg = C['THZCudaTensor_arg']
local THZCudaTensor_norm = C['THZCudaTensor_norm']

local THZCudaTensor_real = C['THZCudaTensor_real']
local THZCudaTensor_imag = C['THZCudaTensor_imag']

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
local THZCudaTensor_cadd = C['THZCudaTensor_cadd']
local THZCudaTensor_cpow = C['THZCudaTensor_cpow']
local THZCudaTensor_cdiv = C['THZCudaTensor_cdiv']

-- wrap("mul",
--      cname("mul"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--         {name=Tensor, method={default=1}},
--         {name=real}}
-- )
--
-- wrap("div",
--      cname("div"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--         {name=Tensor, method={default=1}},
--         {name=real}})
--
-- for _, name in ipairs({"cmul", "cpow", "cdiv"}) do
--   wrap(name,
--        cname(name),
--        {{name=Tensor, default=true, returned=true, method={default='nil'}},
--           {name=Tensor, method={default=1}},
--         {name=Tensor}})
-- end
--
-- wrap("addcmul",
--      cname("addcmul"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--         {name=Tensor, method={default=1}},
--         {name=real, default=1},
--         {name=Tensor},
--         {name=Tensor}})
--
-- wrap("addcdiv",
--      cname("addcdiv"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--         {name=Tensor, method={default=1}},
--         {name=real, default=1},
--         {name=Tensor},
--         {name=Tensor}})
-- wrap("add",
--      cname("add"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--       {name=Tensor, method={default=1}},
--       {name=real}},
--      cname("cadd"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--       {name=Tensor, method={default=1}},
--       {name=real, default=1},
--       {name=Tensor}},
--     cname("cadd"),
--     {{name=Tensor, default=true, returned=true, method={default='nil'}},
--      {name=Tensor, method={default=1}},
--      {name=real, default=1},
--      {name=Tensor}})
-- local THZCudaTensor_fftShiftedInplace = C['THZCudaTensor_fftShiftedInplace']
-- local THZCudaTensor_fftShifted = C['THZCudaTensor_fftShifted']
-- local THZCudaTensor_fftShiftInplace = C['THZCudaTensor_fftShiftInplace']
-- local THZCudaTensor_fftShift = C['THZCudaTensor_fftShift']
-- local THZCudaTensor_ifftShiftInplace = C['THZCudaTensor_ifftShiftInplace']
-- local THZCudaTensor_ifftShift = C['THZCudaTensor_ifftShift']

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
   {name="dst", type=ctypename},
   {name="src", type=typename},
   overload=ZTensor.abs,
   call = function(dst, src)
      THZCudaTensor_zabs(cutorch._state,dst:cdata(), src:cdata())
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

ZTensor.norm = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = torch.CudaTensor()
      THZCudaTensor_norm(cutorch._state,dst:cdata(), src:cdata())
      return dst
   end
}

ZTensor.norm = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename},
   overload=ZTensor.norm,
   call = function(dst, src)
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

ZTensor.fft = argcheck{
   nonamed=true,
   {name="dst", type=typename, opt=true},
   {name="src1", type=typename},
   call =
      function(dst, src1)
         dst = dst or src1
         THZCudaTensor_fft(cutorch._state,dst:cdata(), src1:cdata())
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
         THZCudaTensor_fftBatched(cutorch._state,dst:cdata(), src1:cdata())
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
         THZCudaTensor_ifftU(cutorch._state,dst:cdata(), src1:cdata())
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
         THZCudaTensor_ifftBatchedU(cutorch._state,dst:cdata(), src1:cdata())
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
         THZCudaTensor_ifftU(cutorch._state,dst:cdata(), src1:cdata())
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
         THZCudaTensor_ifftBatchedU(cutorch._state,dst:cdata(), src1:cdata())
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
         src=src:cdata()
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
         src=src:cdata()
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
         src=src:cdata()
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
         src=src:cdata()
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
         src=src:cdata()
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
         src=src:cdata()
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
         src=src:cdata()
         THZCudaTensor_copyDouble(dst, src)
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
-- wrap("mul",
--      cname("mul"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--         {name=Tensor, method={default=1}},
--         {name=real}}
-- )
--
-- wrap("div",
--      cname("div"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--         {name=Tensor, method={default=1}},
--         {name=real}})
--
-- for _, name in ipairs({"cmul", "cpow", "cdiv"}) do
--   wrap(name,
--        cname(name),
--        {{name=Tensor, default=true, returned=true, method={default='nil'}},
--           {name=Tensor, method={default=1}},
--         {name=Tensor}})
-- end
--
-- wrap("addcmul",
--      cname("addcmul"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--         {name=Tensor, method={default=1}},
--         {name=real, default=1},
--         {name=Tensor},
--         {name=Tensor}})
--
-- wrap("addcdiv",
--      cname("addcdiv"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--         {name=Tensor, method={default=1}},
--         {name=real, default=1},
--         {name=Tensor},
--         {name=Tensor}})
-- wrap("add",
--      cname("add"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--       {name=Tensor, method={default=1}},
--       {name=real}},
--      cname("cadd"),
--      {{name=Tensor, default=true, returned=true, method={default='nil'}},
--       {name=Tensor, method={default=1}},
--       {name=real, default=1},
--       {name=Tensor}},
--     cname("cadd"),
--     {{name=Tensor, default=true, returned=true, method={default='nil'}},
--      {name=Tensor, method={default=1}},
--      {name=real, default=1},
--      {name=Tensor}})

local zmetatable = torch.getmetatable('torch.ZCudaTensor')
rawset( zmetatable, 'type', Tensor__type)
rawset( zmetatable, 'typeAs', Tensor__typeAs)
rawset( zmetatable, 'zfloat', Tensor__zfloat)
rawset( zmetatable, 'norm', ZTensor.norm)
rawset( zmetatable, 'im', ZTensor.im)
rawset( zmetatable, 're', ZTensor.re)
rawset( zmetatable, 'fillIm', ZTensor.fillIm)
rawset( zmetatable, 'fillRe', ZTensor.fillRe)
rawset( zmetatable, 'copyIm', ZTensor.copyIm)
rawset( zmetatable, 'copyRe', ZTensor.copyRe)
rawset( zmetatable, 'arg', ZTensor.arg)
rawset( zmetatable, 'abs', ZTensor.abs)
rawset( zmetatable, 'normAll', ZTensor.normall)
rawset( zmetatable, 'normDim', ZTensor.normDim)
rawset( zmetatable, 'polar', ZTensor.polar)
rawset( zmetatable, 'cim', ZTensor.cim)
rawset( zmetatable, 'cre', ZTensor.cre)
rawset( zmetatable, 'copy', ZTensor.copy)
rawset( zmetatable, 'fft', ZTensor.fft)
rawset( zmetatable, 'ifft', ZTensor.ifft)
rawset( zmetatable, 'ifftU', ZTensor.ifftU)
rawset( zmetatable, 'fftBatched', ZTensor.fftBatched)
rawset( zmetatable, 'ifftBatched', ZTensor.ifftBatched)
rawset( zmetatable, 'ifftBatchedU', ZTensor.ifftBatchedU)
rawset( zmetatable, 'add', ZTensor.add)
rawset( zmetatable, 'addcmul', ZTensor.addcmul)
rawset( zmetatable, 'addcdiv', ZTensor.addcdiv)

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

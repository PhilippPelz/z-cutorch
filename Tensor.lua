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

local THZCudaTensor_polar = C['THZCudaTensor_polar']
local THZCudaTensor_cim = C['THZCudaTensor_cim']
local THZCudaTensor_cre = C['THZCudaTensor_cre']

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
  print('im here')
  local zfloat = self:type('torch.ZFloatTensor')
  print('im here')
  return Tensor__type(zfloat,"torch.ZCudaTensor")
end

local function ZFTensor__zcuda(self)
  return torch.ZCudaTensor(self:size()):copy(self:cdata())
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

ZTensor.abs = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = CudaTensor.new()
      THZCudaTensor_abs(cutorch._state,dst:cdata(), src)
      return dst
   end
}
ZTensor.abs = argcheck{
   nonamed=true,
   {name="dst", type=ctypename},
   {name="src", type=typename},
   overload=ZTensor.abs,
   call = function(dst, src)
      THZCudaTensor_zabs(cutorch._state,dst, src)
      return dst
   end
}

ZTensor.arg = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = CudaTensor.new()
      THZCudaTensor_arg(cutorch._state,dst:cdata(), src)
      return dst
   end
}

ZTensor.arg = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename},
   overload=ZTensor.arg,
   call = function(dst, src)
      THZCudaTensor_zarg(cutorch._state,dst, src)
      return dst
   end
}

ZTensor.re = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = CudaTensor.new()
      THZCudaTensor_real(cutorch._state,dst:cdata(), src)
      return dst
   end
}
ZTensor.re = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename},
   overload=ZTensor.re,
   call = function(dst, src)
      THZCudaTensor_zreal(cutorch._state,dst, src)
      return dst
   end
}

ZTensor.im = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = CudaTensor.new()
      THZCudaTensor_imag(cutorch._state,dst:cdata(), src)
      return dst
   end
}
ZTensor.im = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename},
   overload=ZTensor.im,
   call = function(dst, src)
      THZCudaTensor_zimag(cutorch._state,dst, src)
      return dst
   end
}

ZTensor.norm = argcheck{
   nonamed=true,
   {name="src", type=typename},
   call = function(src)
      local dst = CudaTensor.new()
      THZCudaTensor_norm(cutorch._state,dst:cdata(), src)
      return dst
   end
}

ZTensor.norm = argcheck{
   nonamed=true,
   {name="dst", type=typename},
   {name="src", type=typename},
   overload=ZTensor.norm,
   call = function(dst, src)
      THZCudaTensor_znorm(cutorch._state,dst, src)
      return dst
   end
}

ZTensor.normall = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="value", type="number"},
   call = function(self, value)
      return THZCudaTensor_normall(cutorch._state,self, value)
   end
}

ZTensor.normDim = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src", type=typename},
   {name="value", type="number"},
   {name="dimension", type="number"},
   call = function(self, src, value, dimension)
      THZCudaTensor_normDim(cutorch._state,self, src, value, dimension)
      return self
   end
}

ZTensor.polar = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src1", type=ctypename},
   {name="src2", type=ctypename},
   call = function(self, src1, src2)
      THZCudaTensor_polar(cutorch._state,self, src1, src2)
      return self
   end
}

ZTensor.cim = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src2", type=ctypename},
   call = function(self, src2)
      THZCudaTensor_cim(cutorch._state,self, self, src2)
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
      THZCudaTensor_cim(cutorch._state,self, src1, src2)
      return self
   end
}

ZTensor.cre = argcheck{
   nonamed=true,
   {name="self", type=typename},
   {name="src2", type=ctypename},
   call = function(self, src2)
      THZCudaTensor_cre(cutorch._state,self, self, src2)
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
      THZCudaTensor_cre(cutorch._state,self, src1, src2)
      return self
   end
}

ZTensor.copy = argcheck{
   nonamed=true,
   name = "copy",
   {name="dst", type=typename},
   {name="src", type=typename},
   call =
      function(dst, src)
         THZCudaTensor_copy(cutorch._state,dst:cdata(), src)
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
         THZCudaTensor_copyByte(cutorch._state,dst:cdata(), src)
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
         THZCudaTensor_copyChar(cutorch._state,dst:cdata(), src)
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
         THZCudaTensor_copyShort(cutorch._state,dst:cdata(), src)
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
         THZCudaTensor_copyInt(cutorch._state,dst:cdata(), src)
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
         THZCudaTensor_copyLong(cutorch._state,dst:cdata(), src)
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
         THZCudaTensor_copyFloat(cutorch._state,dst:cdata(), src)
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

local zmetatable = torch.getmetatable('torch.ZCudaTensor')
rawset( zmetatable, 'type', Tensor__type)
rawset( zmetatable, 'typeAs', Tensor__typeAs)
rawset( zmetatable, 'zfloat', Tensor__zfloat)
rawset( zmetatable, 'norm', ZTensor.norm)
rawset( zmetatable, 'im', ZTensor.im)
rawset( zmetatable, 're', ZTensor.re)
rawset( zmetatable, 'arg', ZTensor.arg)
rawset( zmetatable, 'abs', ZTensor.abs)
rawset( zmetatable, 'normAll', ZTensor.normall)
rawset( zmetatable, 'normDim', ZTensor.normDim)
rawset( zmetatable, 'polar', ZTensor.polar)
rawset( zmetatable, 'cim', ZTensor.cim)
rawset( zmetatable, 'cre', ZTensor.cre)
rawset( zmetatable, 'copy', ZTensor.copy)

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

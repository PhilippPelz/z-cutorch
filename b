Warning: unmatched variable LUALIB
-- The C compiler identification is GNU 4.9.2
-- The CXX compiler identification is GNU 4.9.2
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Found Torch7 in /home/philipp/torch/install
-- Found CUDA: /usr/local/cuda (found suitable version "7.5", minimum required is "5.5") 
-- Compiling with MAGMA support

Error: Build error: Failed building.
cmake -E make_directory build && cd build && cmake .. -DLUALIB= -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/home/philipp/torch/install/bin/.." -DCMAKE_INSTALL_PREFIX="/home/philipp/torch/install/lib/luarocks/rocks/zcutorch/scm-1" -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda && make -j$(getconf _NPROCESSORS_ONLN) install


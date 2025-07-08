FROM debian:stable

RUN apt-get update --fix-missing
RUN apt-get install -y wget
RUN apt-get install -y emacs
RUN apt-get install -y git git-lfs
RUN apt-get install -y build-essential gfortran
RUN apt-get install -y clang clang-format
RUN apt-get install -y cmake cmake-curses-gui
RUN apt-get install -y libopenblas-dev liblapack-dev liblapacke-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libmetis-dev
RUN apt-get install -y libscotchparmetis-dev
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y xorg-dev
RUN apt-get install -y libglfw3-dev
RUN apt-get install -y libnlopt-dev libnlopt-cxx-dev
RUN apt-get install -y libpython3-dev

RUN cd && echo "\"\e[A\": history-search-backward" >> .inputrc
RUN cd && echo "\"\e[B\": history-search-forward" >> .inputrc

# OCC & ESP
RUN cd $HOME; wget https://acdl.mit.edu/ESP/OCC770-linux-x86_64.tgz; tar -xvf OCC770-linux-x86_64.tgz

ENV CASROOT="/root/OpenCASCADE-7.7.0"
ENV CAS_DIR="$CASROOT"

RUN cd $HOME; wget https://acdl.mit.edu/ESP/ESP.tgz; tar -xvf ESP.tgz

ENV ESP_ROOT="/root/EngSketchPad"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CAS_DIR/lib"

RUN cd $ESP_ROOT/config; ./makeEnv
RUN cd $ESP_ROOT/; . $ESP_ROOT/ESPenv.sh; cd src; make -j

# avro
RUN cd $HOME; git clone --recursive https://gitlab.com/philipclaude/avro.git; 
RUN cd $HOME/avro; mkdir build; cd build; . $ESP_ROOT/ESPenv.sh; ESP_DIR=$ESP_ROOT cmake .. -DCMAKE_BUILD_TYPE=Release; CPATH=$CPATH:$ESP_ROOT/include make -j avro

# refine
RUN cd $HOME; git clone https://github.com/nasa/refine.git; 
RUN cd $HOME/refine; mkdir build; cd build; . $ESP_ROOT/ESPenv.sh; cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CAS_DIR;$ESP_ROOT"; make -j install

# mmg
RUN cd $HOME; git clone https://github.com/MmgTools/mmg.git
RUN cd $HOME/mmg; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release; make -j install

# omega_h
RUN cd $HOME; git clone https://github.com/LoicMarechal/libMeshb.git
RUN cd $HOME/libMeshb; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr; make -j install
RUN cd $HOME; git clone https://github.com/jeromerobert/omega_h.git; cd omega_h; git checkout develop
RUN cd $HOME/omega_h; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DOmega_h_USE_libMeshb=ON; make -j install

ENV PATH="/root/avro/build/bin:$PATH"
ENV LD_LIBRARY_PATH="/root/avro/build/lib:$ESP_ROOT/lib:$CASROOT/lib:$LD_LIBRARY_PATH"
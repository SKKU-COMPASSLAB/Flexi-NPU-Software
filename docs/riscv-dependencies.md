# RISC-V Dependencies

기존에 제공하던 [TDS Simulator Spike Extension](https://github.com/SKKU-COMPASSLAB/TDS-Simulator-Spike-Extension)의 경우 RISC-V 의존성 소프트웨어들을 submodule로 제공하였으나, 최근 해당 레포지토리들에 대한 미러 서버 접근과 관련된 이슈가 발생하여 현재는 특정 버전을 submodule로 제공하지 않는다. 본 시뮬레이터 및 소프트웨어를 사용하는 사용자는 `externals` 디렉터리에 별도로 RISC-V GNU 툴체인 및 Spike ISS를 설치하거나, 기존에 설치된 경로를 `RISCV` 환경변수를 통해 세팅하여 올바르게 모든 요소들이 컴파일 될 수 있도록 해야 한다.

## RISC-V GNU Toolchain

```bash
mkdir -p externals
export RISCV=$PWD/externals/riscv-gnu-toolchain/build

###################################################
# Install RISC-V GNU Toolchain
###################################################

# STEP 1: Install dependencies
sudo apt-get install autoconf automake autotools-dev curl python3 python3-pip python3-tomli libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev libslirp-dev libncurses-dev

# STEP 2: Download repository
git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git externals/riscv-gnu-toolchain
cd externals/riscv-gnu-toolchain

git fetch origin --tags
git checkout tags/2026.07.15                # currently, we tested 2026.07.15-nightly version
git submodule update --init --recursive     # it takes a little for a while to download all submodules ...

# STEP 3: Install toolchain
./configure --prefix="$RISCV" --with-arch=rv64gcv --with-abi=lp64d
make -j
```

## RISC-V Spike ISS

```bash
mkdir -p externals
export RISCV=$PWD/externals/riscv-gnu-toolchain/build

###################################################
# Install RISC-V Spike ISS
###################################################

# STEP 1: Download repository
git clone https://github.com/riscv-software-src/riscv-isa-sim.git externals/riscv-isa-sim
cd externals/riscv-isa-sim

# STEP 2: Build simulator
mkdir -p build
cd build
../configure --prefix="$RISCV"
make -j

# STEP 3: Install simulator
make install
```
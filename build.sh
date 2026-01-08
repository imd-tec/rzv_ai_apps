. /opt/poky/3.33/environment-setup-aarch64-poky-linux
export SDK=/opt/poky/3.1.33/
cd 05_Age_gender_detection/src/
mkdir -p build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=./toolchain/runtime.cmake -DV2H=ON ..
make -j 24
scp age_gender_detection_app root@172.16.30.100:/usr/bin/
# Remove existing build directory if it exists
if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi
mkdir build
cd build
cmake ..
make -j

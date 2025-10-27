#!/bin/bash

# Usage: ./build.sh [--no-tcmalloc] [--use-aio] [--search-only]
# Options:
#   --no-tcmalloc    Disable tcmalloc, use system default malloc
#   --use-aio        Use AIO instead of liburing
#   --search-only    Enable search-only mode (READ_ONLY_TESTS and NO_MAPPING)

# Default options
USE_TCMALLOC=ON
USE_AIO=OFF
SEARCH_ONLY=OFF

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --no-tcmalloc)
            USE_TCMALLOC=OFF
            echo "Option: Disabling tcmalloc, using system malloc"
            ;;
        --use-aio)
            USE_AIO=ON
            echo "Option: Using AIO instead of liburing"
            ;;
        --search-only)
            SEARCH_ONLY=ON
            echo "Option: Enabling search-only mode (READ_ONLY_TESTS and NO_MAPPING)"
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-tcmalloc    Disable tcmalloc, use system default malloc"
            echo "  --use-aio        Use AIO instead of liburing"
            echo "  --search-only    Enable search-only mode (READ_ONLY_TESTS and NO_MAPPING)"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Remove existing build directory if it exists
if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi

mkdir build
cd build

# Run cmake with selected options
echo "Building with: USE_TCMALLOC=$USE_TCMALLOC USE_AIO=$USE_AIO SEARCH_ONLY=$SEARCH_ONLY"
cmake -DUSE_TCMALLOC=$USE_TCMALLOC -DUSE_AIO=$USE_AIO -DSEARCH_ONLY=$SEARCH_ONLY ..

# Build
make -j

#!/bin/bash

# Isaac Gym 설치 스크립트
# 여러 위치에서 Isaac Gym을 찾아서 현재 conda 환경에 설치합니다

echo "Searching for Isaac Gym installations..."

# 가능한 Isaac Gym 경로들
ISAAC_PATHS=(
    "/home/kyungminlee/PBHC/isaacgym"
    "/home/kyungminlee/OpenHL/isaacgym"
    "/home/kyungminlee/ProtoMotions/isaacgym"
)

FOUND_PATH=""

# 가장 최근에 수정된 경로 찾기
for path in "${ISAAC_PATHS[@]}"; do
    if [ -f "$path/python/setup.py" ]; then
        if [ -z "$FOUND_PATH" ] || [ "$path/python/setup.py" -nt "$FOUND_PATH/python/setup.py" ]; then
            FOUND_PATH="$path"
        fi
    fi
done

if [ -z "$FOUND_PATH" ]; then
    echo "Error: Isaac Gym not found in any of the expected locations."
    echo "Please install Isaac Gym first or specify the path manually."
    exit 1
fi

echo "Found Isaac Gym at: $FOUND_PATH"
echo "Installing Isaac Gym to current conda environment..."

cd "$FOUND_PATH/python"
pip install -e .

if [ $? -eq 0 ]; then
    echo "✓ Isaac Gym installed successfully!"
    echo "You can verify with: python -c 'import isaacgym; print(isaacgym.__file__)'"
else
    echo "✗ Isaac Gym installation failed!"
    exit 1
fi


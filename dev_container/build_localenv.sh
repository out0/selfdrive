#! /bin/bash
rm -rf ../venv/
rm -rf venv/

check_virtualenv() {
    if ! command -v virtualenv &> /dev/null; then
        echo "virtualenv is not installed. Installing..."
        python3 -m pip install --user virtualenv
        echo "virtualenv installation complete."
    fi
}

create_venv() {
    # Check if virtualenv is installed, if not, install it
    check_virtualenv
    
    local env_name=${1:-".venv"}

    if [ -d "$env_name" ]; then
        echo "Virtual environment '$env_name' already exists. Aborting."
        return 1
    fi

    python3 -m venv "$env_name"
    source "./$env_name/bin/activate"
    pip install -U pip
}

activate_venv() {
    local env_name=${1:-".venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create one."
        return 1
    fi

    source "./$env_name/bin/activate"
}

install_deps() {
    local env_name=${1:-".venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create one."
        return 1
    fi

    source "./$env_name/bin/activate"

    if [ -f "requirements.txt" ]; then
        pip install -r ./requirements.txt
    fi

    if [ -f "setup.py" ]; then
        pip install -e .
    fi
}

install_my_modules() {
    local env_name=${1:-".venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create one."
        return 1
    fi

    source "./$env_name/bin/activate"

    cd ../libdriveless/python_bind && pip3 install -e .
    cd ../../libgpd/python_bind && pip3 install -e .
    cd ../../carla_driver && pip3 install -e .
}

create_venv "../venv"
install_deps "../venv"
install_my_modules "../venv"
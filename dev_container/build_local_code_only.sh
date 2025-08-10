#! /bin/bash
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

install_my_modules "../venv"
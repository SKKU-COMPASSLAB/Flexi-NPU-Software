#!/usr/bin/env bash

_flexi_env_status=0
if [ -z "${RISCV:-}" ]; then
    printf 'flexi env error: RISCV is not set. Set RISCV to the RISC-V toolchain and Spike installation prefix.\n' >&2
    _flexi_env_status=1
elif [ ! -d "${RISCV}" ]; then
    printf 'flexi env error: RISCV does not name a directory: %s\n' "${RISCV}" >&2
    _flexi_env_status=1
elif [ ! -d "${RISCV}/bin" ]; then
    printf 'flexi env error: RISCV/bin does not exist: %s/bin\n' "${RISCV}" >&2
    _flexi_env_status=1
fi

_flexi_required_tools="riscv64-unknown-elf-gcc riscv64-unknown-elf-ar riscv64-unknown-elf-objdump spike"
if [ "${_flexi_env_status}" -eq 0 ]; then
    for _flexi_tool in ${_flexi_required_tools}; do
        if [ ! -x "${RISCV}/bin/${_flexi_tool}" ]; then
            printf 'flexi env error: required executable is missing: %s/bin/%s\n' "${RISCV}" "${_flexi_tool}" >&2
            _flexi_env_status=1
        fi
    done
    if [ ! -d "${RISCV}/include" ]; then
        printf 'flexi env error: Spike headers are missing: %s/include\n' "${RISCV}" >&2
        _flexi_env_status=1
    fi
    if [ ! -d "${RISCV}/lib" ]; then
        printf 'flexi env error: Spike library directory is missing: %s/lib\n' "${RISCV}" >&2
        _flexi_env_status=1
    fi
fi

if [ "${_flexi_env_status}" -ne 0 ]; then
    unset _flexi_env_status
    unset _flexi_required_tools
    unset _flexi_tool
    return 1 2>/dev/null || exit 1
fi

case ":${PATH}:" in
    *":${RISCV}/bin:"*) ;;
    *) export PATH="${RISCV}/bin:${PATH}" ;;
esac

unset _flexi_required_tools
unset _flexi_tool
unset _flexi_env_status

printf 'flexi environment ready: RISCV=%s\n' "${RISCV}"

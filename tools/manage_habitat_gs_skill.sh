#!/usr/bin/env bash
set -euo pipefail

# Skills shipped by this repo (under skills/<name>/).
ALL_SKILLS=("habitat-gs-control" "habitat-gs-train")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  tools/manage_habitat_gs_skill.sh install   --workspace PATH [--skill NAME] [--mode copy|symlink] [--force]
  tools/manage_habitat_gs_skill.sh uninstall --workspace PATH [--skill NAME] [--force]
  tools/manage_habitat_gs_skill.sh status    --workspace PATH [--skill NAME]

Skills:
  habitat-gs-control   Interactively pilot a robot in a live sim via MCP tools
  habitat-gs-train     Train and evaluate a navigation policy (PointNav/ImageNav/ObjectNav, VLN)

Notes:
  - Operates on skill paths: <workspace>/skills/<name>
  - Without --skill, ALL shipped skills are processed
  - Default install mode is "copy"
  - "symlink" mode requires the target path to be accessible to the agent process
EOF
}

log() {
    printf '[skill-manager] %s\n' "$*"
}

die() {
    printf '[skill-manager][error] %s\n' "$*" >&2
    exit 1
}

# is_managed_target <skill_name> <target_dir>
is_managed_target() {
    local skill_name="$1"
    local target_dir="$2"
    local source_skill_dir="${REPO_ROOT}/skills/${skill_name}"

    if [[ -L "${target_dir}" ]]; then
        local resolved
        resolved="$(readlink -f "${target_dir}" || true)"
        [[ "${resolved}" == "${source_skill_dir}" ]] && return 0
    fi

    if [[ -f "${target_dir}/SKILL.md" ]] && rg -q "^name:\s*${skill_name}\b" "${target_dir}/SKILL.md"; then
        return 0
    fi

    return 1
}

# install_skill <skill_name> <workspace> <mode> <force>
install_skill() {
    local skill_name="$1"
    local workspace="$2"
    local mode="$3"
    local force="$4"
    local source_skill_dir="${REPO_ROOT}/skills/${skill_name}"
    local target_dir="${workspace}/skills/${skill_name}"

    [[ -d "${source_skill_dir}" ]] || die "Source skill not found: ${source_skill_dir}"
    mkdir -p "${workspace}/skills"

    if [[ -e "${target_dir}" || -L "${target_dir}" ]]; then
        if [[ "${force}" != "1" ]] && ! is_managed_target "${skill_name}" "${target_dir}"; then
            die "Refusing to overwrite unmanaged target: ${target_dir} (use --force to override)"
        fi
        rm -rf "${target_dir}"
    fi

    if [[ "${mode}" == "symlink" ]]; then
        ln -s "${source_skill_dir}" "${target_dir}"
        log "Installed ${skill_name} via symlink: ${target_dir} -> ${source_skill_dir}"
    else
        mkdir -p "${target_dir}"
        cp -a "${source_skill_dir}/." "${target_dir}/"
        chmod +x "${target_dir}/scripts/hab" 2>/dev/null || true
        log "Installed ${skill_name} via copy: ${target_dir}"
    fi

    if [[ -f "${target_dir}/.env.example" ]]; then
        log "  Skill env template: ${target_dir}/.env.example"
    fi
    return 0
}

# uninstall_skill <skill_name> <workspace> <force>
uninstall_skill() {
    local skill_name="$1"
    local workspace="$2"
    local force="$3"
    local target_dir="${workspace}/skills/${skill_name}"

    if [[ ! -e "${target_dir}" && ! -L "${target_dir}" ]]; then
        log "Nothing to remove: ${target_dir}"
        return
    fi

    if [[ "${force}" != "1" ]] && ! is_managed_target "${skill_name}" "${target_dir}"; then
        die "Refusing to remove unmanaged target: ${target_dir} (use --force to override)"
    fi

    rm -rf "${target_dir}"
    log "Removed ${target_dir}"
}

# status_skill <skill_name> <workspace>
status_skill() {
    local skill_name="$1"
    local workspace="$2"
    local source_skill_dir="${REPO_ROOT}/skills/${skill_name}"
    local target_dir="${workspace}/skills/${skill_name}"
    local skill_env="${target_dir}/.env"

    echo "[${skill_name}]"
    echo "  source_skill=${source_skill_dir}"
    echo "  target=${target_dir}"

    if [[ -L "${target_dir}" ]]; then
        echo "  target_type=symlink"
        echo "  target_resolved=$(readlink -f "${target_dir}" || true)"
    elif [[ -d "${target_dir}" ]]; then
        echo "  target_type=directory"
        echo "  skill_installed=true"
    else
        echo "  target_type=missing"
    fi

    echo "  skill_env_exists=$([[ -f "${skill_env}" ]] && echo yes || echo no)"
}

main() {
    [[ $# -ge 1 ]] || { usage; exit 1; }

    local action="$1"
    shift

    local workspace=""
    local skill=""
    local mode="copy"
    local force="0"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --workspace)
                [[ $# -ge 2 ]] || die "--workspace requires a value"
                workspace="$2"
                shift 2
                ;;
            --skill)
                [[ $# -ge 2 ]] || die "--skill requires a value"
                skill="$2"
                shift 2
                ;;
            --mode)
                [[ $# -ge 2 ]] || die "--mode requires a value"
                mode="$2"
                shift 2
                ;;
            --force)
                force="1"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
    done

    [[ "${mode}" == "copy" || "${mode}" == "symlink" ]] || die "--mode must be copy or symlink"
    [[ -n "${workspace}" ]] || die "Provide --workspace PATH"

    workspace="$(readlink -f "${workspace}")"
    [[ -d "${workspace}" ]] || die "Workspace path does not exist: ${workspace}"

    local skills=()
    if [[ -z "${skill}" ]]; then
        skills=("${ALL_SKILLS[@]}")
    else
        local known=0 s2
        for s2 in "${ALL_SKILLS[@]}"; do
            [[ "${s2}" == "${skill}" ]] && { known=1; break; }
        done
        [[ "${known}" == "1" ]] || die "Unknown --skill '${skill}'. Known skills: ${ALL_SKILLS[*]}"
        skills=("${skill}")
    fi

    local s
    case "${action}" in
        install)
            for s in "${skills[@]}"; do install_skill "${s}" "${workspace}" "${mode}" "${force}"; done
            ;;
        uninstall)
            for s in "${skills[@]}"; do uninstall_skill "${s}" "${workspace}" "${force}"; done
            ;;
        status)
            for s in "${skills[@]}"; do status_skill "${s}" "${workspace}"; done
            ;;
        *)
            die "Unknown action: ${action}"
            ;;
    esac
}

main "$@"

#!/bin/bash
# Vision Project Agent
# 管理 /home/duang/vision 项目的自动化agent

set -e

PROJECT_DIR="/home/duang/vision"
DOCKER_CONTAINER="robot_vision"
STATUS_DOC="$PROJECT_DIR/docs/agent-status.md"
HELPER_SCRIPT="$PROJECT_DIR/agent_helper.py"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 更新状态文档
update_status() {
    local task="$1"
    local status="$2"
    local details="$3"

    python3 "$HELPER_SCRIPT" update-status "$task" "$status" "$details"
}

# 在Docker容器中编译
build_in_docker() {
    log_info "开始在Docker容器中编译..."
    update_status "编译项目" "进行中" "使用colcon build编译ROS2项目"

    if docker exec "$DOCKER_CONTAINER" bash -c "cd /workspace && colcon build --symlink-install"; then
        log_info "编译成功"
        update_status "编译项目" "完成" "编译成功，无错误"
        return 0
    else
        log_error "编译失败"
        update_status "编译项目" "失败" "编译出错，请检查日志"
        return 1
    fi
}

# 运行测试
run_tests() {
    log_info "运行测试..."
    update_status "运行测试" "进行中" "执行单元测试"

    if docker exec "$DOCKER_CONTAINER" bash -c "cd /workspace && colcon test"; then
        log_info "测试通过"
        update_status "运行测试" "完成" "所有测试通过"
        return 0
    else
        log_warn "测试失败"
        update_status "运行测试" "失败" "部分测试未通过"
        return 1
    fi
}

# 分析代码变更
analyze_changes() {
    log_info "分析代码变更..."
    python3 "$HELPER_SCRIPT" analyze-changes
}

# 生成代码摘要
generate_summary() {
    log_info "生成代码摘要..."
    python3 "$HELPER_SCRIPT" generate-summary
}

# 执行自定义命令
execute_command() {
    local cmd="$1"
    log_info "执行命令: $cmd"
    update_status "执行命令" "进行中" "$cmd"

    if docker exec "$DOCKER_CONTAINER" bash -c "cd /workspace && $cmd"; then
        log_info "命令执行成功"
        update_status "执行命令" "完成" "$cmd 执行成功"
        return 0
    else
        log_error "命令执行失败"
        update_status "执行命令" "失败" "$cmd 执行失败"
        return 1
    fi
}

# 显示帮助
show_help() {
    cat << EOF
Vision Project Agent - 使用说明

命令:
  build           - 在Docker容器中编译项目
  test            - 运行测试
  analyze         - 分析代码变更
  summary         - 生成代码摘要
  exec <cmd>      - 在Docker容器中执行自定义命令
  status          - 显示当前状态
  help            - 显示此帮助信息

示例:
  ./agent.sh build
  ./agent.sh exec "ros2 run realsense_subscriber realsense_subscriber_node"
  ./agent.sh analyze
EOF
}

# 显示状态
show_status() {
    if [ -f "$STATUS_DOC" ]; then
        cat "$STATUS_DOC"
    else
        log_warn "状态文档不存在"
    fi
}

# 主函数
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    case "$1" in
        build)
            build_in_docker
            ;;
        test)
            run_tests
            ;;
        analyze)
            analyze_changes
            ;;
        summary)
            generate_summary
            ;;
        exec)
            if [ $# -lt 2 ]; then
                log_error "请提供要执行的命令"
                exit 1
            fi
            shift
            execute_command "$*"
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

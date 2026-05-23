#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Project Agent Helper
辅助工具：代码分析、状态更新、摘要生成
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path("/home/duang/vision")
STATUS_DOC = PROJECT_DIR / "docs" / "agent-status.md"
DOCS_DIR = PROJECT_DIR / "docs"


class AgentHelper:
    def __init__(self):
        self.project_dir = PROJECT_DIR
        self.status_doc = STATUS_DOC
        self.docs_dir = DOCS_DIR

    def update_status(self, task, status, details):
        """更新状态文档"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 读取现有状态
        if self.status_doc.exists():
            with open(self.status_doc, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = self._create_initial_status()

        # 添加新状态
        new_entry = f"\n### {timestamp} - {task}\n"
        new_entry += f"**状态**: {status}\n"
        new_entry += f"**详情**: {details}\n"

        # 插入到任务记录部分
        if "## 任务记录" in content:
            parts = content.split("## 任务记录")
            content = parts[0] + "## 任务记录" + new_entry + "\n" + parts[1]
        else:
            content += "\n## 任务记录" + new_entry

        # 写入文件
        with open(self.status_doc, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ 状态已更新: {task} - {status}")

    def _create_initial_status(self):
        """创建初始状态文档"""
        return """# Agent 状态报告

> 最近更新: {timestamp}

## 当前任务

无进行中的任务

## 项目概况

- **项目路径**: /home/duang/vision
- **Docker容器**: robot_vision
- **主要模块**: realsense_subscriber

## 任务记录

""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def analyze_changes(self):
        """分析代码变更"""
        print("分析代码变更...")

        try:
            # 获取git状态
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.project_dir,
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                changes = result.stdout.strip().split('\n')
                print(f"发现 {len(changes)} 个文件变更:")
                for change in changes:
                    print(f"  {change}")

                # 更新状态文档
                self.update_status(
                    "代码变更分析",
                    "完成",
                    f"发现 {len(changes)} 个文件变更"
                )
            else:
                print("没有检测到代码变更")
                self.update_status(
                    "代码变更分析",
                    "完成",
                    "工作区干净，无变更"
                )
        except Exception as e:
            print(f"✗ 分析失败: {e}")
            self.update_status("代码变更分析", "失败", str(e))

    def generate_summary(self):
        """生成代码摘要"""
        print("生成代码摘要...")

        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modules": [],
            "files_count": 0,
            "lines_count": 0
        }

        # 扫描src目录
        src_dir = self.project_dir / "src"
        if src_dir.exists():
            for module_dir in src_dir.iterdir():
                if module_dir.is_dir() and not module_dir.name.startswith('.'):
                    module_info = self._analyze_module(module_dir)
                    summary["modules"].append(module_info)
                    summary["files_count"] += module_info["files_count"]
                    summary["lines_count"] += module_info["lines_count"]

        # 生成摘要文档
        self._write_summary(summary)
        print(f"✓ 摘要已生成: {summary['files_count']} 个文件, {summary['lines_count']} 行代码")

    def _analyze_module(self, module_dir):
        """分析单个模块"""
        module_info = {
            "name": module_dir.name,
            "files_count": 0,
            "lines_count": 0,
            "cpp_files": [],
            "hpp_files": []
        }

        # 统计代码文件
        for ext in ['.cpp', '.cc', '.cxx']:
            cpp_files = list(module_dir.rglob(f'*{ext}'))
            module_info["cpp_files"].extend([f.name for f in cpp_files])
            module_info["files_count"] += len(cpp_files)

            for f in cpp_files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        module_info["lines_count"] += len(file.readlines())
                except:
                    pass

        for ext in ['.hpp', '.h', '.hxx']:
            hpp_files = list(module_dir.rglob(f'*{ext}'))
            module_info["hpp_files"].extend([f.name for f in hpp_files])
            module_info["files_count"] += len(hpp_files)

            for f in hpp_files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        module_info["lines_count"] += len(file.readlines())
                except:
                    pass

        return module_info

    def _write_summary(self, summary):
        """写入摘要文档"""
        summary_doc = self.docs_dir / "code-summary.md"

        content = f"""# 代码摘要

> 生成时间: {summary['timestamp']}

## 统计信息

- **总文件数**: {summary['files_count']}
- **总代码行数**: {summary['lines_count']}
- **模块数量**: {len(summary['modules'])}

## 模块详情

"""

        for module in summary['modules']:
            content += f"\n### {module['name']}\n\n"
            content += f"- 文件数: {module['files_count']}\n"
            content += f"- 代码行数: {module['lines_count']}\n"

            if module['cpp_files']:
                content += f"- C++源文件: {', '.join(module['cpp_files'])}\n"
            if module['hpp_files']:
                content += f"- 头文件: {', '.join(module['hpp_files'])}\n"

        with open(summary_doc, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    if len(sys.argv) < 2:
        print("用法: agent_helper.py <command> [args...]")
        print("命令:")
        print("  update-status <task> <status> <details>")
        print("  analyze-changes")
        print("  generate-summary")
        sys.exit(1)

    helper = AgentHelper()
    command = sys.argv[1]

    if command == "update-status":
        if len(sys.argv) < 5:
            print("错误: update-status 需要3个参数")
            sys.exit(1)
        helper.update_status(sys.argv[2], sys.argv[3], sys.argv[4])

    elif command == "analyze-changes":
        helper.analyze_changes()

    elif command == "generate-summary":
        helper.generate_summary()

    else:
        print(f"未知命令: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

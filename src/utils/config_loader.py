"""
配置加载模块

功能：
1. 加载 YAML 配置文件
2. 提供配置访问接口
3. 支持配置验证和默认值
"""

import yaml
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: str = "config/corner_detection.yaml"):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径（相对于项目根目录）
        """
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        self.current_profile: str = "default"

        # 加载配置文件
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        full_path = os.path.join(project_root, self.config_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"配置文件不存在: {full_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            self.config_data = yaml.safe_load(f)

        # 验证配置文件
        self._validate_config()

    def _validate_config(self):
        """验证配置文件格式"""
        required_profiles = ['default']
        for profile in required_profiles:
            if profile not in self.config_data:
                raise ValueError(f"配置文件缺少必需的配置项: {profile}")

        # 验证默认配置的必需字段
        required_sections = ['angle_computation', 'gradient', 'clustering',
                            'inner_corner', 'algorithm']
        default_config = self.config_data['default']

        for section in required_sections:
            if section not in default_config:
                raise ValueError(f"默认配置缺少必需的节: {section}")

    def set_profile(self, profile: str):
        """
        设置当前配置文件

        Args:
            profile: 配置文件名称（default, high_precision, fast, experimental）
        """
        if profile not in self.config_data:
            raise ValueError(f"配置文件不存在: {profile}")

        self.current_profile = profile

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的路径）

        Args:
            key_path: 配置键路径，如 "gradient.threshold"
            default: 默认值（如果键不存在）

        Returns:
            配置值

        Example:
            >>> config = ConfigLoader()
            >>> threshold = config.get("gradient.threshold")
            >>> print(threshold)  # 3.0
        """
        keys = key_path.split('.')
        value = self.config_data.get(self.current_profile, {})

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

            if value is None:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取整个配置节

        Args:
            section: 配置节名称，如 "gradient"

        Returns:
            配置节字典
        """
        profile_config = self.config_data.get(self.current_profile, {})
        return profile_config.get(section, {})

    def get_all(self) -> Dict[str, Any]:
        """
        获取当前配置文件的所有配置

        Returns:
            完整配置字典
        """
        return self.config_data.get(self.current_profile, {})

    def list_profiles(self) -> list:
        """
        列出所有可用的配置文件

        Returns:
            配置文件名称列表
        """
        return list(self.config_data.keys())


# 全局配置实例（单例模式）
_global_config: Optional[ConfigLoader] = None


def get_config(config_path: str = "config/corner_detection.yaml") -> ConfigLoader:
    """
    获取全局配置实例（单例模式）

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigLoader 实例
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigLoader(config_path)

    return _global_config


def reload_config(config_path: str = "config/corner_detection.yaml"):
    """
    重新加载配置文件

    Args:
        config_path: 配置文件路径
    """
    global _global_config
    _global_config = ConfigLoader(config_path)


if __name__ == "__main__":
    # 测试配置加载
    config = ConfigLoader()

    print("可用配置文件:", config.list_profiles())
    print("\n默认配置:")
    print(f"  梯度阈值: {config.get('gradient.threshold')}")
    print(f"  高斯滤波标准差: {config.get('angle_computation.gaussian_sigma')}")
    print(f"  最小角点间隔比例: {config.get('clustering.min_interval_ratio')}")

    print("\n切换到高精度配置:")
    config.set_profile('high_precision')
    print(f"  梯度阈值: {config.get('gradient.threshold')}")
    print(f"  高斯滤波标准差: {config.get('angle_computation.gaussian_sigma')}")

    print("\n切换到快速配置:")
    config.set_profile('fast')
    print(f"  梯度阈值: {config.get('gradient.threshold')}")
    print(f"  高斯滤波标准差: {config.get('angle_computation.gaussian_sigma')}")

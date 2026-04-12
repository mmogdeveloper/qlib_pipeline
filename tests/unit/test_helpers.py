"""
工具函数测试
覆盖股票代码转换、日期解析、配置加载
"""

import pytest
from datetime import datetime


class TestStockCodeConversion:
    """股票代码转换测试"""

    def test_stock_code_to_qlib_sh_prefix(self):
        """600000 → SH600000"""
        from utils.helpers import stock_code_to_qlib

        assert stock_code_to_qlib("600000") == "SH600000"

    def test_stock_code_to_qlib_sz_prefix(self):
        """000001 → SZ000001"""
        from utils.helpers import stock_code_to_qlib

        assert stock_code_to_qlib("000001") == "SZ000001"

    def test_stock_code_to_qlib_beijing(self):
        """688001（科创板）→ SH688001"""
        from utils.helpers import stock_code_to_qlib

        assert stock_code_to_qlib("688001") == "SH688001"

    def test_stock_code_to_qlib_9_prefix(self):
        """9 开头 → SH"""
        from utils.helpers import stock_code_to_qlib

        assert stock_code_to_qlib("900001") == "SH900001"

    def test_stock_code_to_qlib_3_prefix(self):
        """300001（创业板）→ SZ300001"""
        from utils.helpers import stock_code_to_qlib

        assert stock_code_to_qlib("300001") == "SZ300001"

    def test_stock_code_to_qlib_pads_short_code(self):
        """短代码应补零"""
        from utils.helpers import stock_code_to_qlib

        assert stock_code_to_qlib("1") == "SZ000001"

    def test_qlib_code_to_akshare_roundtrip(self):
        """往返转换保持一致"""
        from utils.helpers import stock_code_to_qlib, qlib_code_to_akshare

        for code in ["600000", "000001", "300001", "688001"]:
            qlib_code = stock_code_to_qlib(code)
            back = qlib_code_to_akshare(qlib_code)
            assert back == code, f"往返失败: {code} → {qlib_code} → {back}"


class TestDateParsing:
    """日期解析测试"""

    def test_str_to_date_dash_format(self):
        """支持 YYYY-MM-DD"""
        from utils.helpers import str_to_date

        dt = str_to_date("2024-01-01")
        assert dt == datetime(2024, 1, 1)

    def test_str_to_date_compact_format(self):
        """支持 YYYYMMDD"""
        from utils.helpers import str_to_date

        dt = str_to_date("20240101")
        assert dt == datetime(2024, 1, 1)

    def test_str_to_date_invalid_raises(self):
        """无法解析的日期应抛 ValueError"""
        from utils.helpers import str_to_date

        with pytest.raises(ValueError, match="无法解析"):
            str_to_date("01-01-2024")

    def test_date_to_str(self):
        """日期对象转字符串"""
        from utils.helpers import date_to_str

        assert date_to_str(datetime(2024, 1, 1)) == "2024-01-01"

    def test_date_to_str_passthrough(self):
        """字符串输入直接返回"""
        from utils.helpers import date_to_str

        assert date_to_str("2024-01-01") == "2024-01-01"


class TestLoadConfig:
    """配置加载测试"""

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        """配置文件不存在应抛 FileNotFoundError"""
        from utils.helpers import load_config
        import utils.helpers as helpers_module

        monkeypatch.setattr(helpers_module, "PROJECT_ROOT", tmp_path)
        (tmp_path / "config").mkdir()

        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            load_config("nonexistent.yaml")

    def test_load_valid_config(self, tmp_path, monkeypatch):
        """加载合法配置文件"""
        from utils.helpers import load_config
        import utils.helpers as helpers_module

        monkeypatch.setattr(helpers_module, "PROJECT_ROOT", tmp_path)
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "test.yaml").write_text("key: value\n", encoding="utf-8")

        result = load_config("test.yaml")
        assert result == {"key": "value"}


class TestMergeConfig:
    """配置合并测试"""

    def test_merge_config_with_args(self):
        """命令行参数应覆盖配置"""
        from utils.helpers import merge_config_with_args

        config = {"a": 1, "b": 2}
        args = {"a": 10, "b": None, "c": 3}
        result = merge_config_with_args(config, args)
        assert result["a"] == 10  # 被覆盖
        assert result["b"] == 2   # None 不覆盖
        assert result["c"] == 3   # 新增

    def test_merge_does_not_mutate_original(self):
        """合并不应修改原字典"""
        from utils.helpers import merge_config_with_args

        config = {"a": 1}
        merge_config_with_args(config, {"a": 2})
        assert config["a"] == 1

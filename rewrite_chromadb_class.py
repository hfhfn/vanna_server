import os
import shutil
import time
import logging
from typing import Optional, Dict, Any
from vanna.chromadb import ChromaDB_VectorStore


class AutoCleanChromaDB(ChromaDB_VectorStore):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        低负载优化的自动清理ChromaDB客户端

        参数配置:
            config = {
                "path": "./chroma_data",               # 数据存储路径
                "on_disk_persistence_auto_clean": True, # 磁盘自动清理
                "max_retention_hours": 24 * 3,        # 数据保留72小时（3天）
                "auto_clean": False,                  # 默认关闭内存自动清理
                "clean_interval": 12 * 3600,          # 低负载下12小时清理一次
                "health_check_interval": 24 * 3600,   # 低负载下每天检查一次
            }
        """
        # 合并默认配置（针对低负载优化）
        default_config = {
            "on_disk_persistence_auto_clean": True,  # 磁盘自动清理
            "max_retention_hours": 24 * 3,  # 保留3天数据
            "auto_clean": False,  # 默认关闭内存自动清理
            "clean_interval": 12 * 3600,  # 12小时清理一次
            "health_check_interval": 24 * 3600,  # 24小时检查一次
            "max_repair_attempts": 2,  # 最大尝试次数， 最大2表示不做删库修复
            "backup_before_repair": False,  # 是否在修复前备份数据
        }
        self.config = {**default_config, **(config or {})}

        # 设置低负载友好的日志级别
        self.logger = logging.getLogger("AutoCleanChromaDB_LowLoad")
        self.logger.setLevel(logging.WARNING)  # 只记录警告及以上级别

        # 初始化父类
        try:
            super().__init__(config=config)
            self.check_db_health(mode="full")
        except Exception as e:
            self.logger.warning(f"初始化失败，尝试修复: {str(e)}")
            self._repair_chroma_db()

        # 启动后台维护（低频率模式）
        self._start_low_load_maintenance()

    def check_db_health(self, mode: str = "lightweight"):
        """
        统一的健康检查方法（兼容 ChromaDB v0.6.0+）

        Args:
            mode: "lightweight" - 快速检查（默认）
                  "full" - 完整完整性检查
        Returns:
            bool: 健康状态
        """
        try:
            # 第一阶段：基础集合存在性检查（兼容新版API）
            try:
                # v0.6.0+ 返回名称列表，旧版返回Collection对象
                collections = set(self.chroma_client.list_collections())  # 直接获取名称集合
            except TypeError:
                # 兼容旧版处理方式
                collections = {c.name for c in self.chroma_client.list_collections()}

            required_collections = {"sql", "ddl", "documentation"}
            missing = required_collections - collections

            if missing:
                self.logger.warning(f"缺失集合: {missing}")
                return False

            # 第二阶段：按模式分层检查
            if mode == "full":
                # 完整模式：验证数据可访问性
                for name in required_collections:
                    try:
                        collection = self.chroma_client.get_collection(name)
                        # 使用peek替代count，避免全表扫描
                        records = collection.peek(limit=1)
                        if not records.get("ids"):
                            self.logger.info(f"集合 {name} 为空（正常状态）")
                    except Exception as e:
                        self.logger.error(f"集合 {name} 访问失败: {str(e)}")
                        return False
            else:
                # 轻量模式：仅验证客户端可访问
                try:
                    # 简单测试一个关键集合
                    self.chroma_client.get_collection("sql").peek(limit=0)
                except Exception as e:
                    self.logger.warning(f"轻量检查异常: {str(e)}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"健康检查异常: {str(e)}", exc_info=True)
            return False

    def _repair_chroma_db(self):
        """渐进式修复策略"""
        repair_attempt = 0

        while repair_attempt < self.config["max_repair_attempts"]:
            repair_attempt += 1
            try:
                if repair_attempt == 1:
                    self._conservative_repair()
                elif repair_attempt == 2:
                    self._moderate_repair()
                else:
                    # chromadb类内部暂时不做激进修复（删除整个数据库），在类外部实现激进修复，最大重试次数设为2不走这个分支。
                    self._aggressive_repair()

                # 验证修复结果
                if self.check_db_health(mode="full"):
                    self.logger.info(f"修复成功 (阶段 {repair_attempt})")
                    return

            except Exception as e:
                self.logger.error(f"修复阶段 {repair_attempt} 失败: {str(e)}")

        raise RuntimeError("所有修复尝试均失败")

    def _conservative_repair(self):
        """保守修复：仅重置客户端连接"""
        self.logger.warning("尝试保守修复...")

        # 1. 尝试重新创建客户端
        if hasattr(self, 'chroma_client'):
            del self.chroma_client
        time.sleep(1)  # 等待资源释放

        # 2. 重新初始化
        super().__init__(config=self.config)

    def _moderate_repair(self):
        """中等修复：重建索引但保留数据"""
        self.logger.warning("尝试中等修复...")

        # 1. 备份重要数据
        if self.config["backup_before_repair"]:
            self._backup_data()

        # 2. 重置集合但不删除数据文件
        for name in ["sql", "ddl", "documentation"]:
            try:
                collection = getattr(self, f"{name}_collection", None)
                if collection:
                    collection.delete(where={})
                    self.logger.info(f"已重置集合: {name}")
            except Exception as e:
                self.logger.warning(f"重置集合 {name} 失败: {str(e)}")

        # 3. 重建客户端
        self._conservative_repair()

    def _aggressive_repair(self):
        """激进修复：完全重建数据库"""
        self.logger.warning("尝试激进修复...")

        # 1. 备份数据
        if self.config["backup_before_repair"]:
            self._backup_data()

        # 2. 完全重置
        db_path = self.config.get("path")
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            os.makedirs(db_path)

        # 3. 全新初始化
        super().__init__(config=self.config)

    def _backup_data(self):
        """备份当前数据库"""
        db_path = self.config.get("path")
        if not os.path.exists(db_path):
            return

        backup_path = f"{db_path}_backup_{int(time.time())}"
        try:
            shutil.copytree(db_path, backup_path)
            self.logger.info(f"数据库已备份到: {backup_path}")
        except Exception as e:
            self.logger.error(f"备份失败: {str(e)}")

    def _start_low_load_maintenance(self):
        """启动低频率后台维护"""
        import threading
        self._maintenance_thread = threading.Thread(
            target=self._low_load_maintenance_loop,
            daemon=True
        )
        self._maintenance_thread.start()

    def _low_load_maintenance_loop(self):
        """低负载维护循环"""
        while True:
            try:
                current_time = time.time()

                # 按超大间隔执行健康检查（默认24小时）
                if current_time - getattr(self, "_last_health_check", 0) > self.config["health_check_interval"]:
                    self.check_db_health()
                    self._last_health_check = current_time

                # 按配置间隔执行清理
                if current_time - getattr(self, "_last_clean_time", 0) > self.config["clean_interval"]:
                    self._low_load_clean()
                    self._last_clean_time = current_time

            except Exception as e:
                self.logger.warning(f"维护任务轻微异常: {str(e)}")

            # 低负载下每小时检查一次即可（原版是每分钟）
            time.sleep(3600)

    def _low_load_clean(self):
        """低负载清理（仅处理过期数据）"""
        if not self.config["on_disk_persistence_auto_clean"]:
            return

        cutoff = time.time() - self.config["max_retention_hours"] * 3600

        for name in ["sql", "ddl", "documentation"]:
            try:
                collection = getattr(self, f"{name}_collection")
                # 仅删除明确过期的数据（降低CPU使用）
                collection.delete(
                    where={"created_at": {"$lt": cutoff}},
                    where_document={}  # 不检查文档内容
                )
            except:
                pass  # 低负载下忽略清理错误

    def close(self):
        """线程安全的资源释放方法"""
        # 1. 停止维护线程
        if hasattr(self, "_maintenance_thread"):
            self._maintenance_thread.join(timeout=2)
            del self._maintenance_thread

        # 2. 释放chromadb客户端资源
        if hasattr(self, "chroma_client"):
            try:
                if hasattr(self.chroma_client, "close"):
                    self.chroma_client.close()
            except Exception as e:
                self.logger.warning(f"关闭客户端失败: {str(e)}")

        # 3. 清理其他资源
        if hasattr(self, "logger"):
            for handler in self.logger.handlers[:]:
                handler.close()
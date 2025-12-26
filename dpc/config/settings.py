import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMConfig:
    model_name: str = "gpt-4o"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 3
    retry_delay: int = 2

@dataclass
class PipelineConfig:
    sql_timeout: int = 30
    epsilon: float = 0.05
    max_correction_attempts: int = 3

@dataclass
class DatasetConfig:
    dataset_type: str = "spider"
    data_path: str = "data/spider/dev.json"
    db_root_path: str = "data/spider/database"
    pred_sqls_path: str = "data/spider/pred_sqls.json"
    output_path: str = "dpc_results.json"

@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def load_from_toml(self, file_path: str):
        """Loads and overrides configuration from a TOML file."""
        import toml
        if not os.path.exists(file_path):
            return
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = toml.load(f)
            
        if "llm" in data:
            for k, v in data["llm"].items():
                if hasattr(self.llm, k):
                    setattr(self.llm, k, v)
        
        if "pipeline" in data:
            for k, v in data["pipeline"].items():
                if hasattr(self.pipeline, k):
                    setattr(self.pipeline, k, v)
                    
        if "dataset" in data:
            for k, v in data["dataset"].items():
                if hasattr(self.dataset, k):
                    setattr(self.dataset, k, v)

# Global default config
config = Config()


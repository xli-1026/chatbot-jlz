from pydantic import RedisDsn, AnyHttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    inference_server_url: AnyHttpUrl = "http://localhost:8080"
    isvc_coder_model_uri: AnyHttpUrl = "http://localhost:8081"
    log_level: str = "INFO"
    redis_om_url: RedisDsn = "redis://localhost:6379"
    """This env name (REDIS_OM_URL) is required by redis-om"""
    user_id_header: str = "kubeflow-userid"
    warehouse_uri: str = "mysql+mysqlconnector://root:root@localhost:3307/"


settings = Settings()

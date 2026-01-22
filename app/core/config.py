from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL_DOCKER: str
    DATABASE_URL_LOCAL: str

    # 🔥 DEV FLAG — bypass token validation
    DEV_BYPASS_TOKEN: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def DATABASE_URL(self) -> str:
        """
        Use local DB when running outside Docker.
        """
        return self.DATABASE_URL_LOCAL


settings = Settings()

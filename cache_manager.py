from pydantic import BaseModel


class CacheClearInput(BaseModel):
    cache_types: list
    console_key: str

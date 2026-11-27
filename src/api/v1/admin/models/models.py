from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

from core.model_types import ModelType


class ModelsByExpertiseRead(BaseModel):
    expertise: Annotated[
        ModelType, Field(description="Expertise of the requested models.")
    ]
    models: Annotated[list[str], Field(description="List of Model Names")]


class ModelBase(BaseModel):
    model_name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=255,
            description="Name of the Model, as used by its Provider.",
        ),
    ]
    expertise: Annotated[ModelType, Field(description="Expertise of the Model.")]


class ModelCreate(ModelBase):
    pass


class ModelRead(ModelBase):
    created_date: Annotated[
        datetime, Field(description="When the Model was registered.")
    ]

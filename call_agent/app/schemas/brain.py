from pydantic import BaseModel


class BrainProgress(BaseModel):
    name_confirmed: bool = False
    email_confirmed: bool = False
    phone_confirmed: bool = False
    calibration_completed: bool = False


class BrainOutput(BaseModel):
    progress: BrainProgress
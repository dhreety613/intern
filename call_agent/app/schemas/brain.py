from pydantic import BaseModel


class BrainProgress(BaseModel):
    name_confirmed: bool = False
    email_confirmed: bool = False
    phone_confirmed: bool = False
    calibration_completed: bool = False
    calibration_guidance_offered: bool = False
    calibration_guidance_wanted: bool = False
    calibration_guidance_step: int = 0


class BrainOutput(BaseModel):
    progress: BrainProgress
import uuid
from dataclasses import asdict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.logging import get_logger
from database.session import get_db_session
from database.user_skills import post_user_skill, get_user_skill_by_skill_id, SkillDescription, update_user_skill_by_skill_id

from .models import SkillCreate, SkillCreateRead, SkillRead, SkillUpdate


router = APIRouter()


tags = ["Skills"]


log = get_logger()


@router.post(
    path="/skill",
    tags=tags,
    description="Create a skill with the provided content and store it in the Vector Store.",
    response_model=SkillCreateRead
)
async def create_skill(
    scope: str,
    payload: SkillCreate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session)
) -> SkillCreateRead:
    """Create a new user skill.
    
    This endpoint creates a user skill record in the database and stores the skill description
    in the Vector Store for semantic search capabilities.
    """
    # TODO: In prep for upcoming skills (for Agents, etc.)
    if not scope in ["user_vs_skills"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unsupported scope: {scope}"
        )
    try:
        skill_id = await post_user_skill(
            session=session,
            user_id=user.id,
            name=payload.name,
            description=payload.description,
            skill_text=payload.skill_text,
            scope=scope,
            parameters_schema=payload.parameters_schema
        )
        
        return SkillCreateRead(
            skill_id=skill_id,
            name=payload.name,
            message=f"Skill '{payload.name}' created successfully"
        )
    
    except ValueError as e:
        log.error(f"Value error creating skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        log.error(f"Error creating skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create skill: {str(e)}"
        )


@router.get(
    "/skill",
    tags=tags,
    description="Retrieves a skill's name, description, instructions and parameter schema.",
    response_model=SkillRead
)
async def get_skill(
    skill_id: str,
    scope: str,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session)
) -> SkillRead:
    try:
        if scope == "user_vs_skills":
            skill: SkillDescription = await get_user_skill_by_skill_id(
                session=session,
                user_id=user.id,
                skill_id=uuid.UUID(skill_id)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unsupported scope: '{scope}'"
            )
        return SkillRead(
            **asdict(skill)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to fetch skill: {str(e)}"
        )


@router.patch(
    "/skill",
    tags=tags,
    description=(
        "Updated a skill its instructions. For complete updates to the skill, including its description, "
        "remove the current skill and add a new one."
    ),
    response_model=SkillRead
)
async def patch_skill(
    scope: str,
    payload: SkillUpdate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session)
) -> SkillRead:
    try:
        if scope == "user_vs_skills":
            skill: SkillDescription = await update_user_skill_by_skill_id(
                session=session,
                user_id=user.id,
                skill_id=payload.skill_id,
                instructions=payload.instructions
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unsupported scope: '{scope}'"
            )
        return SkillRead(
            **asdict(skill)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to fetch skill: {str(e)}"
        )
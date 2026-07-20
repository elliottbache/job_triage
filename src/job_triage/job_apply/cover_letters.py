from job_triage.job_apply.schemas import (
    ApplicationProse,
    CoverLetter,
    JobApplicationInfo,
)


def create_cover_letter(
    prose: ApplicationProse, job_application: JobApplicationInfo
) -> CoverLetter:
    """Create structured cover letter content for one job application."""
    return CoverLetter(
        job_id=job_application.job_id,
        greeting="Dear Hiring Team,",
        subject=f"Application for {job_application.title}",
        body=prose.cover_letter_text.strip(),
        signature="Elliott Bache",
    )

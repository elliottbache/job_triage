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


def render_cover_letter_text(cover_letter: CoverLetter) -> str:
    """Render structured cover letter content as plain text."""
    sections = [
        cover_letter.greeting.strip(),
        f"Subject: {cover_letter.subject.strip()}",
        cover_letter.body.strip(),
        f"Sincerely,\n{cover_letter.signature.strip()}",
    ]

    return "\n\n".join(sections) + "\n"

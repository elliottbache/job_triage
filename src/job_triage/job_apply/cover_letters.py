import re
import tomllib
from pathlib import Path

from job_triage._helpers import ROOT_DIR
from job_triage.job_apply.resumes import latex_escape, looks_north_american
from job_triage.job_apply.schemas import (
    ApplicantConfig,
    ApplicationProse,
    CoverLetter,
    JobApplicationInfo,
)

_DEFAULT_APPLICANT_CONFIG_PATH = ROOT_DIR / "applicant.toml"


def read_applicant_config(
    path: Path = _DEFAULT_APPLICANT_CONFIG_PATH,
) -> ApplicantConfig:
    """Read private applicant config from a TOML file."""
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Applicant config not found at {path}. "
            "Copy applicant.example.toml to applicant.toml and fill in your local values."
        ) from exc

    return ApplicantConfig.model_validate(data)


def create_cover_letter(
    prose: ApplicationProse,
    job_application: JobApplicationInfo,
    applicant_config: ApplicantConfig,
) -> CoverLetter:
    """Create structured cover letter content for one job application."""
    is_north_america = looks_north_american(
        job_application, job_application.source_json
    )
    contact = (
        applicant_config.applicant.north_america
        if is_north_america
        else applicant_config.applicant.eu
    )

    return CoverLetter(
        job_id=job_application.job_id,
        first_name=applicant_config.applicant.first_name,
        family_name=applicant_config.applicant.family_name,
        address_line_1=contact.address_line_1,
        address_line_2=contact.address_line_2,
        mobile=contact.mobile,
        email=applicant_config.applicant.email,
        linkedin=applicant_config.applicant.linkedin,
        github=applicant_config.applicant.github,
        recipient_name=applicant_config.cover_letter.recipient_name,
        recipient_address=applicant_config.cover_letter.recipient_address,
        opening=applicant_config.cover_letter.opening,
        closing=applicant_config.cover_letter.closing,
        subject=f"Application for {job_application.title}",
        body=prose.cover_letter_text.strip(),
        signature=applicant_config.applicant.signature,
    )


def render_cover_letter_text(cover_letter: CoverLetter) -> str:
    """Render structured cover letter content as plain text."""
    sections = [
        cover_letter.opening.strip(),
        f"Subject: {cover_letter.subject.strip()}",
        cover_letter.body.strip(),
        f"{cover_letter.closing.strip()}\n{cover_letter.signature.strip()}",
    ]

    return "\n\n".join(sections) + "\n"


def render_cover_letter_tex(cover_letter: CoverLetter) -> str:
    """Render structured cover letter content as a moderncv LaTeX document."""
    body = _render_cover_letter_body_tex(cover_letter.body)
    contact_details = _render_cover_letter_contact_details_tex(cover_letter)
    recipient_name = _render_cover_letter_recipient_name_tex(cover_letter)

    return rf"""\documentclass[11pt,letterpaper,sans]{{moderncv}}
\moderncvstyle{{banking}}
\moderncvcolor{{blue}}
\nopagenumbers{{}}

\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage[scale=0.8]{{geometry}}
\usepackage{{lmodern}}
\recomputelengths

\firstname{{{latex_escape(cover_letter.first_name.strip())}}}
\familyname{{{latex_escape(cover_letter.family_name.strip())}}}
\address{{{latex_escape(cover_letter.address_line_1.strip())}}}{{{latex_escape(cover_letter.address_line_2.strip())}}}
\extrainfo{{{contact_details}}}

\recipient{{{recipient_name}}}{{{latex_escape(cover_letter.recipient_address.strip())}}}
\date{{\today}}
\opening{{{latex_escape(cover_letter.opening.strip())}}}
\closing{{{latex_escape(cover_letter.closing.strip())}}}

\pdfobjcompresslevel=0
\input{{glyphtounicode}}
\pdfgentounicode=1

\begin{{document}}
\makelettertitle

{body}

\makeletterclosing
\end{{document}}
"""


def _render_cover_letter_recipient_name_tex(cover_letter: CoverLetter) -> str:
    """Render the recipient name without moderncv's default bold styling."""
    return rf"\textnormal{{{latex_escape(cover_letter.recipient_name.strip())}}}"


def _render_cover_letter_contact_details_tex(cover_letter: CoverLetter) -> str:
    """Render the contact block with a forced break before social links."""
    mobile = latex_escape(cover_letter.mobile.strip())
    email = latex_escape(cover_letter.email.strip())
    linkedin = latex_escape(cover_letter.linkedin.strip())
    github = latex_escape(cover_letter.github.strip())

    return (
        rf"\begin{{tabular}}{{c}}"
        rf"\phonesymbol~{mobile}\hspace{{2ex}}"
        rf"\emailsymbol~\href{{mailto:{email}}}{{{email}}}"
        rf"\\[0.35em]\linkedinsocialsymbol~"
        rf"\href{{https://www.linkedin.com/in/{linkedin}}}{{{linkedin}}} "
        rf"\hspace{{2ex}}"
        rf"\githubsocialsymbol~\href{{https://github.com/{github}}}{{{github}}}"
        rf"\end{{tabular}}"
    )


def _render_cover_letter_body_tex(body: str) -> str:
    """Render cover letter body paragraphs with LaTeX escaping."""
    paragraphs = [
        " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        for paragraph in re.split(r"\n\s*\n", body.strip())
    ]

    return "\n\n".join(latex_escape(paragraph) for paragraph in paragraphs if paragraph)

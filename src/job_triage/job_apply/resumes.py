import re

from job_triage.job_apply.schemas import ApplicationPlan, JobApplicationInfo


def render_resume_tex(
    plan: ApplicationPlan,
    job_application: JobApplicationInfo,
    *,
    force_north_america: bool | None = None,
) -> str:
    """Render a moderncv resume/CV from an application plan and job application info."""
    full_text = job_application.source_json
    is_north_america = (
        force_north_america
        if force_north_america is not None
        else _looks_north_american(job_application, full_text)
    )
    include_academic = plan.selected_base_resume in {"rse", "cfd"}

    sections = [
        _render_preamble(is_north_america=is_north_america),
        _render_document_start(),
        _render_summary(plan),
        _render_authorization(is_north_america=is_north_america),
        _render_core_skills(plan),
        _render_ai_work(plan, full_text),
        _render_experience(plan),
        _render_projects(plan),
        _render_education(include_academic=include_academic),
    ]

    if include_academic:
        sections.append(_render_patents_publications_conferences())

    sections.extend(
        [
            _render_languages(),
            _render_interests(),
            r"\end{document}",
        ]
    )

    return "\n\n".join(section for section in sections if section.strip()) + "\n"


def _render_preamble(*, is_north_america: bool) -> str:
    """Render the LaTeX preamble and contact block."""
    paper = "letterpaper" if is_north_america else "a4paper"

    if is_north_america:
        address = r"\address{Boynton Beach, FL}{USA}"
        mobile = r"\mobile{(561) 859 3344}"
    else:
        address = r"\address{Valencia}{Spain}"
        mobile = r"\mobile{+34 636 15 78 38}"

    return rf"""\documentclass[{paper},10pt]{{moderncv}}
\moderncvstyle{{classic}}
\moderncvcolor{{green}}
\nopagenumbers{{}}

\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage[scale=0.8]{{geometry}}
\usepackage{{lmodern}}
\usepackage{{enumitem}}
\recomputelengths

\firstname{{Elliott}}
\familyname{{Bache}}
{address}
{mobile}
\email{{elliottbache@gmail.com}}
\social[linkedin]{{elliottbache}}
\social[github]{{elliottbache}}

\newcommand{{\cvtab}}[1]{{%
\noindent\begin{{tabular}}{{@{{}}p{{\hintscolumnwidth}} p{{3mm}} p{{\dimexpr\linewidth-\hintscolumnwidth-3mm\relax}}@{{}}}}
#1
\end{{tabular}}
}}
\setlist[itemize]{{leftmargin=1.15em, itemsep=2pt, parsep=0pt, topsep=2pt}}

\pdfobjcompresslevel=0
\input{{glyphtounicode}}
\pdfgentounicode=1"""


def _render_document_start() -> str:
    """Render the opening document commands."""
    return "\n".join(
        [
            r"\begin{document}",
            r"\makecvtitle",
            r"\vspace{-5mm}",
        ]
    )


def _render_summary(plan: ApplicationPlan) -> str:
    """Render the professional summary section."""
    return rf"""\section{{Professional Summary}}
\cvline{{}}{{{_latex_escape(plan.tailored_summary)}}}"""


def _render_authorization(*, is_north_america: bool) -> str:
    """Render work authorization and location section."""
    if is_north_america:
        authorization = "U.S. citizen; no sponsorship required"
        hours = "U.S./Canada"
    else:
        authorization = "French/EU citizen; authorized to work in the EU"
        hours = "European or U.S."

    return rf"""\section{{Work Authorization \& Location}}
\cvtab{{
Authorization: & & {_latex_escape(authorization)} \\
Location: & & Remote; able to work {_latex_escape(hours)} business hours; willing to relocate temporarily for onboarding \\
}}"""


def _render_core_skills(plan: ApplicationPlan) -> str:
    """Render core skills from plan.core_skills."""
    rows = [
        rf"{_latex_escape(label)}: & & {_latex_escape(skills)} \\"
        for label, skills in plan.core_skills.items()
    ]

    return "\\section{Core Skills}\n\\cvtab{\n" + "\n".join(rows) + "\n}"


def _render_ai_work(plan: ApplicationPlan, full_text: str) -> str:
    """Render fixed AI/LLM work only when the job explicitly mentions AI or LLM."""
    if not _contains_caps_ai_or_llm(full_text):
        return ""

    lines = [
        r"\cvline{AI-assisted dev}{Use AI coding tools in a practical, human-reviewed workflow for brainstorming, code exploration, debugging support, test generation, and faster iteration on implementation ideas.}",
        r"\cvline{Compliance AI}{Integrated Anthropic/Claude into a FastAPI compliance workflow to generate structured site-analysis previews from inspection history, with Pydantic response schemas, evidence-reference validation, retry handling, and Markdown report generation.}",
        r"\cvline{Eval Design}{Built validation-oriented AI workflows with strict schema checks, reference checks against known site-history records, and human-review-only output boundaries to avoid unsupported compliance decisions.}",
        r"\cvline{LLM training}{Train and evaluate AI models on math and physics problem-solving tasks; review outputs for correctness, edge cases, reasoning quality, and explanation clarity.}",
    ]

    return "\\section{AI \\& LLM Work}\n" + "\n".join(lines)


def _render_experience(plan: ApplicationPlan) -> str:
    """Render selected professional experience."""
    entries = []

    for role in plan.selected_experience:
        bullets = "\n".join(
            rf"  \item {_latex_escape(bullet)}" for bullet in role.bullets
        )

        entries.append(
            rf"""\cvline{{{_latex_escape(role.years)}}}{{\textbf{{{_latex_escape(role.company)}}} --- {_latex_escape(role.job_title)}
\begin{{itemize}}
{bullets}
\end{{itemize}}
}}"""
        )

    return "\\section{Professional Experience}\n" + "\n".join(entries)


def _render_projects(plan: ApplicationPlan) -> str:
    """Render selected projects."""
    lines = [
        rf"\cvline{{{_latex_escape(project.label)}}}{{{_latex_escape(project.description)}}}"
        for project in plan.selected_projects
    ]

    return "\\section{Selected Projects}\n" + "\n".join(lines)


def _render_education(*, include_academic: bool) -> str:
    """Render education, including graduate degrees for RSE/CFD versions."""
    lines = []

    if include_academic:
        lines.extend(
            [
                r"\cvline{2011}{\textbf{Ph.D. Aerospace Engineering}, Technical University of Madrid (UPM), Spain}",
                r"\cvline{2006}{\textbf{M.S. Mechanical Engineering / Energetics}, INSA Toulouse, France}",
            ]
        )

    lines.append(
        r"\cvline{}{\textbf{B.S. Mechanical Engineering}, Florida Institute of Technology, USA (GPA 3.81/4.00, \textit{summa cum laude})}"
    )

    return "\\section{Education}\n" + "\n".join(lines)


def _render_patents_publications_conferences() -> str:
    """Render patents, publications, and conferences for RSE/CFD versions."""
    entries = [
        r"Spanish patent ES2567002B1: ``Thermal storage tank at high temperature'' (submitted Oct 2014; issued Apr 2016).",
        r"``Impregnation of composite materials: a numerical study.'' \textit{Applied Composite Materials}, 2017.",
        r"``Effect of domain subdivisions on alloy solidification.'' \textit{Journal of Energy Storage}, 2018.",
        r"``Model reduction in the back step fluid-thermal problem with variable geometry.'' \textit{International Journal of Thermal Sciences}, 2010.",
        r"``Computationally efficient reduced order model to generate multi-parameter fluid-thermal databases.'' \textit{International Journal of Thermal Sciences}, 2012.",
        r"``Off-eutectic binary salt finite volume method.'' SolarPACES 2013; \textit{Energy Procedia} 49 (2014) 715--724.",
        r"``FUROWAKE: a new wake model for estimation of wind farm energy production.'' Wind Energy Science Conf., 2017.",
        r"``Smooth hill validation in FUROW's wind resource module using OpenFOAM.'' 5th Symp. on OpenFOAM in Wind Energy, 2017.",
        r"Airbus PhD Day (2010): CENIT-ICARO optimization topic.",
        r"World Congress on Engineering (2011): model reduction for variable geometry.",
    ]

    lines = [rf"\cvline{{}}{{{entry}}}" for entry in entries]

    return "\\section{Patents, Publications, Conferences}\n" + "\n".join(lines)


def _render_languages() -> str:
    """Render languages section."""
    return r"""\section{Languages}
\cvline{}{English (native), French (fluent), Spanish (fluent), German (beginner), Chinese (basic concepts)}"""


def _render_interests() -> str:
    """Render interests section."""
    return r"""\section{Interests}
\cvline{}{Swimming, camping in RV, piano, reading}"""


def _looks_north_american(job_application: JobApplicationInfo, full_text: str) -> bool:
    """Infer whether US/Canada contact info should be used.

    job_application.location is a Literal.  If it is "Worldwide" or "Other", then we
    must decide if the company is North American or not.  Besides "US", all other locations
    are automatically non-North American.
    """
    if job_application.location in {"US", "Canada"}:
        return True
    if job_application.location not in {"Worldwide", "Other"}:
        return False

    _NORTH_AMERICA_PATTERN = re.compile(
        r"(?<!\w)(?:"
        r"u\.s\.|us|usa|united states|canada|canadian|"
        r"north america|new york|san francisco|toronto|vancouver"
        r")(?!\w)",
        flags=re.IGNORECASE,
    )

    return bool(_NORTH_AMERICA_PATTERN.search(full_text))


def _contains_caps_ai_or_llm(full_text: str) -> bool:
    """Return True when the job post explicitly contains AI or LLM in caps."""
    return bool(re.search(r"\b(?:AI|LLM)\b", full_text))


def _latex_escape(text: str) -> str:
    """Escape user/LLM-generated text for LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    return "".join(replacements.get(char, char) for char in str(text))


if __name__ == "__main__":
    pass

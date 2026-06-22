from job_triage.job_apply.schemas import ApplicationJobPost, ResumeContext

resume_context = ResumeContext(
    post=ApplicationJobPost(
        title="Thermal CFD Engineer",
        job_description=(
            "We need a CFD engineer for thermal-fluid and multiphase simulation "
            "work. The role includes ANSYS Fluent, heat transfer, solidification, "
            "thermal storage, reduced-order modeling, validation against "
            "experimental data, parametric studies, HPC execution, Python "
            "post-processing, and clear technical documentation."
        ),
        metadata_text={
            "location": "Remote / EU",
            "employment_type": "FullTime",
            "work_arrangement": "Remote",
            "source_url": "fixture://cfd_thermal_multiphase_engineer",
        },
    ),
    stack_mentions=[
        "ansys fluent",
        "thermal cfd",
        "heat transfer",
        "solidification",
        "multiphase cfd",
        "reduced-order modeling",
        "python post-processing",
        "hpc",
        "validation",
        "technical documentation",
    ],
)

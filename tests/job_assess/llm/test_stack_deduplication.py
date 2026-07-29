from job_triage.job_assess.llm.stack_deduplication import deduplicate_stack_mentions


class TestDeduplicateStackMentions:
    def test_uses_shared_skill_deduplication_for_mentions(
        self, stack_mention_factory
    ) -> None:
        mentions = [
            stack_mention_factory(skill="Python", substitutes=["Ruby"]),
            stack_mention_factory(skill="python", substitutes=["Go"]),
        ]

        result = deduplicate_stack_mentions(mentions)

        assert len(result) == 1
        assert result[0].skill == "Python"
        assert result[0].substitutes == ["Ruby", "Go"]

#############################################################################
# File: episodic_agent.py
#
# Description:
#   Answers questions from fast episodic memory by reconstructing a focused
#   timeline from source-derived or structured event records. It keeps strong
#   evidence from being crowded out by chronology while limiting retrieval and
#   prompt work that does not materially improve the answer.
#
#   - Uses smaller candidate pools and event neighborhoods while retaining extra
#     breadth for temporal, multi-event, update, ecommerce, and EventQA questions.
#   - Keeps strong retrieval recall for temporal, multi-event, update, conflict,
#     recommendation, and EventQA questions without adding extra LLM calls.
#   - Prefers current state for update-style questions while preserving historical
#     state when the question explicitly asks for it.
#   - Selects the most relevant records before sorting them into source order, so
#     early low-value neighbors cannot consume the answer context budget.
#   - Uses exact source anchors when hybrid retrieval surfaces wording that event
#     extraction may have compressed.
#   - Uses cheap character estimates for relevance preselection and leaves exact
#     token-safe packing to the existing context packer.
#   - Omits empty metadata and retrieval-only scores from the answer prompt.
#   - Reduces exact global source timelines when they exceed the context budget.
#   - Gives the answer model explicit rules for timestamps, source order, updates,
#     contradictions, attribution, abstention, and multi-event reasoning.
#   - Prints compact stage updates and a 60-second generation heartbeat so slow API
#     responses remain visibly active without flooding the terminal.
#############################################################################

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from asdrp.agent.BaseAgent import AgentAnswer, BaseAgent
from asdrp.eval_schemas import EvaluationQuestion, RetrievalPlan, RetrievedMemory
from asdrp.memory.episodic_memory import EpisodicMemoryBlock


class EpisodicAgent(BaseAgent):
    # Rebuild answers from stored events, exact source evidence, and their order.
    #
    # Retrieval decides which evidence matters first. The final context then restores
    # source order so chronology helps reasoning without letting the oldest retrieved
    # records automatically take the whole token budget.

    memory_block_type = EpisodicMemoryBlock  # Stores each source event as an episode.

    async def answer(
        self,
        question: EvaluationQuestion,
        default_plan: RetrievalPlan,
    ) -> AgentAnswer:
        # Retrieve event neighborhoods and reason over their temporal sequence.

        plan, strategy = self._select_retrieval_plan(question, default_plan)
        memories, retrieval_seconds = await self._retrieve(question.text, plan)
        print(
            f"[episodic] retrieved | memories={len(memories):,} | "
            f"retrieval={retrieval_seconds:.2f}s | preparing",
            flush=True,
        )

        preparation_started = time.perf_counter()  # Starts the context timer.
        if plan.mode in {"all", "global"}:
            ordered_context_memories = sorted(memories, key=self._timeline_sort_key)
            context, context_tokens = await self._reduce_global_context(
                question=question.text,
                memories=ordered_context_memories,
                final_token_budget=plan.max_context_tokens,
                formatter=self._format_episode,
                system_prompt=(
                    "Compress episodic memory into a concise question-relevant "
                    "timeline. Use only the supplied memory."
                ),
                reduction_instruction=(
                    "Keep only evidence that can answer or disambiguate the question. "
                    "Preserve exact names, numbers, dates, IDs, negation, ownership, "
                    "constraints, corrections, conflicts, and relevant state changes. "
                    "Keep timestamps distinct from source order. Do not infer facts. "
                    "Be concise."
                ),
            )
        else:
            # Select by relevance before chronology. This fixes the common failure where
            # a large neighbor window returns many early records and pack_context stops
            # before reaching a later, much stronger seed.
            context_memories = self._select_context_memories(
                memories,
                max_tokens=plan.max_context_tokens,
            )
            ordered_context_memories = sorted(
                context_memories,
                key=self._timeline_sort_key,
            )
            context, context_tokens = self._pack_context(
                ordered_context_memories,
                max_tokens=plan.max_context_tokens,
                formatter=self._format_episode,
            )
        preparation_seconds = time.perf_counter() - preparation_started

        historical_question = self._is_historical_question(question)
        current_state_rule = (
            "When records give successive states for the same fact, entity, or "
            "preference, use the latest supported state for the current answer. Do not "
            "let a later unrelated event overwrite an earlier fact."
            if plan.prefer_latest and not historical_question
            else (
                "Use the state that was true at the time requested. Preserve superseded "
                "facts when the question asks about an earlier point in the timeline."
            )
        )

        system_prompt = f"""
            You are the reasoning component of an EPISODIC MEMORY agent.
            
            Use only the supplied memory. Reconstruct only the timeline needed for the question.
            Keep different people, objects, sessions, and occasions separate.
            
            Rules:
            - {current_state_rule}
            - Explicit dates/times control chronology when clear; otherwise use source order.
            - For corrections or conflicts, choose the state requested by the question.
            - For before/after/next/first/last, find the anchor event before choosing its neighbor.
            - Combine records only when the evidence connects them; preserve attribution and negation.
            - Prefer exact source detail over a consistent summary when both are present.
            - If memory does not establish the answer, say it is not present rather than guessing.
            - Return only the final answer in the format requested by the question.
        """.strip()

        user_prompt = f"""
            QUESTION TYPE: {question.metadata.get("question_type") or "unspecified"}
            
            RETRIEVED EPISODIC TIMELINE:
            {context or "[No relevant episodes were retrieved]"}
            
            QUESTION:
            {question.text}
        """.strip()

        print(
            f"[episodic] context ready | tokens={context_tokens:,} | generating",
            flush=True,
        )
        generation_started = time.perf_counter()
        generation_task = asyncio.create_task(
            self._generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        )
        try:
            while True:
                done, _ = await asyncio.wait({generation_task}, timeout=60.0)
                if done:
                    hypothesis, generation_seconds = generation_task.result()
                    break
                print(
                    f"[episodic] generation still running | "
                    f"elapsed={time.perf_counter() - generation_started:.0f}s",
                    flush=True,
                )
        except BaseException:
            generation_task.cancel()
            await asyncio.gather(generation_task, return_exceptions=True)
            raise

        # retrieved_count still measures search output, not only the subset that fit in
        # the final prompt, so evaluator metrics keep their original meaning.
        return self._answer_result(
            hypothesis=hypothesis,
            memories=memories,
            context_tokens=context_tokens,
            retrieval_seconds=retrieval_seconds,
            preparation_seconds=preparation_seconds,
            generation_seconds=generation_seconds,
            plan=plan,
            strategy=strategy,
        )

    def _select_retrieval_plan(
        self,
        question: EvaluationQuestion,
        default_plan: RetrievalPlan,
    ) -> tuple[RetrievalPlan, str]:
        # Spend retrieval breadth where it improves recall, not on extra model calls.

        if default_plan.mode in {"all", "global"}:
            return (
                default_plan,
                "exact complete episodic timeline with recursive reduction",
            )

        question_type = self._question_type(question)
        top_k = max(
            default_plan.top_k, 6
        )  # Keep a small recall margin for simple queries.
        neighbor_window = default_plan.neighbor_window
        candidate_multiplier = max(default_plan.candidate_multiplier, 3)
        prefer_latest = default_plan.prefer_latest
        reasons = [f"hybrid top_k={top_k}"]

        # Abstention questions benefit from enough evidence to verify absence without
        # flooding the answer with unrelated records.
        if question_type == "abstention":
            top_k = max(default_plan.top_k, 6)
            reasons.append("controlled abstention breadth")

        if self._is_temporal_question(question):
            top_k = max(top_k, 12)
            neighbor_window = max(neighbor_window, 1)
            candidate_multiplier = max(candidate_multiplier, 4)
            reasons.append("wide temporal candidate pool")

        if self._is_multi_hop_question(question):
            top_k = max(top_k, 14)
            neighbor_window = max(neighbor_window, 1)
            candidate_multiplier = max(candidate_multiplier, 4)
            reasons.append("multi-event candidate pool")

        # These benchmark labels normally ask which state wins after an update or
        # conflict. Historical wording still overrides this in the answer prompt.
        if question_type in {
            "knowledge-update",
            "knowledge_update",
            "temporal_update",
            "conflict_resolution",
        }:
            prefer_latest = True
            top_k = max(top_k, 12)
            neighbor_window = max(neighbor_window, 1)
            reasons.append("current-state resolution")

        if question_type in {
            "recommendation_from_memory",
            "preference_recall",
            "constraint_recall",
            "purchase_history_recall",
            "cart_or_wishlist_recall",
            "return_support_recall",
        }:
            top_k = max(top_k, 8)
            candidate_multiplier = max(candidate_multiplier, 4)
            reasons.append("exact ecommerce recall")

        if self.source.casefold().startswith("eventqa"):
            top_k = max(top_k, 14)
            neighbor_window = max(neighbor_window, 2)
            candidate_multiplier = max(candidate_multiplier, 4)
            reasons.append("fine-grained EventQA sequence")

        return (
            replace(
                default_plan,
                top_k=top_k,
                neighbor_window=neighbor_window,
                candidate_multiplier=candidate_multiplier,
                prefer_latest=prefer_latest,
            ),
            "; ".join(reasons),
        )

    def _select_context_memories(
        self,
        memories: list[RetrievedMemory],
        *,
        max_tokens: int,
    ) -> list[RetrievedMemory]:
        # Keep the highest-value records that fit, then let the caller restore chronology.

        if not memories or max_tokens <= 0:
            return []

        # A cheap character estimate is enough for relevance preselection. _pack_context
        # performs the final exact token-safe packing, so repeated tokenizer calls here only
        # add CPU time without changing correctness.
        rendered_sizes = {
            memory.memory_id: max(1, (len(self._format_episode(memory, 1)) + 3) // 4)
            for memory in memories
        }
        total_tokens = sum(rendered_sizes.values()) + max(0, len(memories) - 1) * 6
        if total_tokens <= max_tokens:
            return list(memories)

        # Seed and exact-source records carry the strongest fused scores. Neighbor
        # episodes have deliberately smaller scores, so they fill only remaining space.
        ranked = sorted(
            memories,
            key=lambda item: (
                item.score,
                self._kind_priority(item),
                item.ordinal,
            ),
            reverse=True,
        )

        selected: list[RetrievedMemory] = []
        used_tokens = 0
        for memory in ranked:
            size = rendered_sizes[memory.memory_id]
            required = size + (6 if selected else 0)

            if used_tokens + required <= max_tokens:
                selected.append(memory)
                used_tokens += required
                continue

            # If the single best record is oversized, keep it so BaseAgent can safely
            # truncate that one record instead of returning no evidence at all.
            if not selected:
                selected.append(memory)
                break

        return selected

    @staticmethod
    def _timeline_sort_key(memory: RetrievedMemory) -> tuple[int, int, float, str]:
        # Exact source evidence comes before derived events from the same source position.

        source_first = 0 if memory.metadata.get("entry_kind") == "source_episode" else 1
        return (memory.ordinal, source_first, -memory.score, memory.memory_id)

    @staticmethod
    def _kind_priority(memory: RetrievedMemory) -> int:
        # Structured events are compact, while exact source anchors are recall fallbacks.

        return 1 if memory.metadata.get("entry_kind") == "episode" else 0

    @staticmethod
    def _format_episode(memory: RetrievedMemory, index: int) -> str:
        # Format exact source anchors and derived event records without duplicating fields.

        metadata = memory.metadata
        entry_kind = str(metadata.get("entry_kind") or "episode")

        if entry_kind == "source_episode":
            timestamp = metadata.get("timestamp") or "unspecified"
            time_line = f"\nTime: {timestamp}" if timestamp != "unspecified" else ""
            return (
                f"[SOURCE {index} | order={memory.ordinal}]\n"
                f"{memory.text}{time_line}"
            )

        participants = metadata.get("participants") or []
        participants_text = (
            ", ".join(map(str, participants)) if participants else "unspecified"
        )
        if not metadata.get("event"):
            # Fast local episodes already contain their exact source text and timestamp in
            # the retrieval text, so avoid storing and formatting a second full-text copy.
            return f"[EPISODE {index} | order={memory.ordinal}]\n{memory.text}"

        event = str(metadata["event"]).strip()
        lines = [f"[EPISODE {index} | order={memory.ordinal}]", f"Event: {event}"]
        if participants_text != "unspecified":
            lines.append(f"Participants: {participants_text}")
        timestamp = metadata.get("time") or metadata.get("timestamp")
        if timestamp:
            lines.append(f"Time: {timestamp}")
        if metadata.get("location"):
            lines.append(f"Location: {metadata['location']}")
        if metadata.get("outcome"):
            lines.append(f"Outcome: {metadata['outcome']}")
        if metadata.get("causal_context"):
            lines.append(f"Cause: {metadata['causal_context']}")
        return "\n".join(lines)

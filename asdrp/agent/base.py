#############################################################################
# base.py
#
# base class for agent replies
#
# @author Theodore Mui
# @email  theodoremui@gmail.com
# Fri Jul 04 11:30:53 PDT 2025
#############################################################################

from dataclasses import dataclass

@dataclass
class AgentReply:
    response_str: str



@dataclass
class AgentBase:


    @property
    def can_batch(self):
        return False
    
    @property
    def batch_all(self):
        return False
    

    def reset_meory(self):
        pass

    def create(cls,**kwargs):
        return cls(**kwargs)

    async def load_chat_history(self, haystack_sessions):
        """
        Replay chat history into the agent's memory block.
        Each session contains turns of user and assistant messages.

        Args:
            agent_object: The agent instance whose memory is populated
            haystack_sessions (list[list[dict]]): List of chat sessions
        """

        print(f"Running {self}...")
        print(f"Processing {len(haystack_sessions)} haystack sessions...")

        memory_block = self.memory_block

        can_batch = self.can_batch
        # can_batch = (
        #     isinstance(self, ReductiveAgent)
        #     or isinstance(self, EpisodicAgent)
        #     or isinstance(self, HVMAgent)
        # )  # ReductiveAgent can accept batched user-assistant pairs without its quality being negatively affected
        # batch_all = isinstance(self, HVMAgent)
        batch_all = self.batch_all
        all_messages = []  # Used only if batch_all is True
        session_count = 0
        turn_count = 0
        for session in haystack_sessions:
            session_count += 1
            if session_count % 5 == 0:  # Print progress every 5 sessions
                print(f"Processed {session_count}/{len(haystack_sessions)} sessions...")

            msg = None  # Content from either user or assistance
            pending_user = None  # Temporary storage to ensure user and assistant content is added together
            buffer: List[ChatMessage] = []  # FIFO buffer queue
            batch_pairs = len(
                session
            )  # Sets each batch size to the maximum number of pairs in a session (1 LLM-call per session, 500 session per question)

            for turn in session:
                turn_count += 1
                content = turn["content"].replace(
                    "<|endoftext|>", ""
                )  # Clean content to avoid tokenizer special-token errors

                # Separate user and assistant messages
                if turn["role"] == "user":
                    msg = ChatMessage(role="user", content=content)
                elif turn["role"] == "assistant":
                    msg = ChatMessage(role="assistant", content=content)
                else:
                    msg = None
                if batch_all:
                    if msg is not None:
                        all_messages.append(msg)
                else:
                    if can_batch:
                        buffer.append(
                            msg
                        )  # Batch user+assistant messages into buffer and flush when full, reduces LLM-calls

                        # Every pair is 2 messages; flush when we reach batch_pairs pairs
                        if len(buffer) >= 2 * batch_pairs:
                            await _retry_aput(memory_block, buffer)
                            buffer = []

                    else:  # Ensures only user-assistant pairs are sent
                        if msg.role == "user":
                            pending_user = msg
                        else:
                            if pending_user is None:
                                continue
                            await _retry_aput(memory_block, [pending_user, msg])

                            # Reset temporary variables for next user-assistant pair
                            pending_user = None
                            msg = None

            # End of session: flush any leftover buffered pairs for batched memory blocks
            if can_batch and buffer and not batch_all:
                await _retry_aput(memory_block, buffer)
        if batch_all and all_messages:
            print(
                f"Flushing all {len(all_messages)} messages (total turns: {turn_count}) to memory in one batch..."
            )
            await _retry_aput(memory_block, all_messages)



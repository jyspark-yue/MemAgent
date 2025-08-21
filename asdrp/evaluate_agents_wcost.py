import json
import asyncio
from asdrp.agent.summary_agent import SummaryAgent
import time
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.openai import OpenAI
import os

# To track cost
INPUT_COST_PER_1K  = 0.0003
OUTPUT_COST_PER_1K = 0.0006

class LongMemEvalRunner:
    def __init__(self, agent, callback_manager, handler):
        self.agent = create_agent(agent, callback_manager)
        self.handler = handler

    async def process_chat_history(self, haystack_sessions, question):
        print(f"Processing {len(haystack_sessions)} haystack sessions...")

        session_count = 0
        for session in haystack_sessions:
            session_count += 1
            if session_count % 5 == 0:  # Print progress every 5 sessions
                print(f"Processed {session_count}/{len(haystack_sessions)} sessions...")

            for turn in session:
                if turn["role"] == "user":
                    try:
                        await self.agent.achat(turn["content"])
                    except Exception as e:
                        print(f"Error processing turn: {e}")
                        continue

        print("Processing final question...")
        response = await self.agent.achat(question)
        return response.response_str

    def reset_memory(self):
        self.agent.memory = self.agent._create_memory()
        self.agent.agent = self.agent._create_agent(self.agent.memory, [])
   

    async def evaluate_on_dataset(self, data_file, output_file, num_questions, start_index=0):
        print(f"Loading dataset from {data_file}...")
        with open(data_file, 'r') as f:
            dataset = json.load(f)

        print(f"Dataset loaded with {len(dataset)} total questions")

        if num_questions is not None:
            dataset = dataset[start_index:start_index+num_questions]
            print(f"Processing questions [{start_index}:{start_index + num_questions}] "
              f"(total this run: {len(dataset)})")
        else:
            dataset = dataset[start_index:]
            print(f"Processing questions [{start_index}:] (total this run: {len(dataset)})")

        results = []

        # To track time
        overall_t0 = time.time()
        overall_prompt0 = self.handler.prompt_llm_token_count
        overall_comp0   = self.handler.completion_llm_token_count

        for i, item in enumerate(dataset):
            print(f"Processing question {i+1}/{len(dataset)}: {item.get('question_id', 'unknown')}")

            # Check the structure of the item
            print(f"Item keys: {list(item.keys())}")
            print(f"Number of haystack sessions: {len(item.get('haystack_sessions', []))}")

            q_t0 = time.time()
            q_prompt0 = self.handler.prompt_llm_token_count
            q_comp0   = self.handler.completion_llm_token_count

            self.reset_memory()
            print("Memory reset, processing chat history...")

            try:
                response = await self.process_chat_history(item['haystack_sessions'], item['question'])
                print(f"Got response: {response[:100]}...")  # Show first 100 chars
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                response = f"Error: {str(e)}"

            results.append({
                'question_id': item['question_id'], 
                'hypothesis': response
            })

            q_prompt = self.handler.prompt_llm_token_count     - q_prompt0
            q_comp   = self.handler.completion_llm_token_count - q_comp0
            q_time   = time.time() - q_t0
            q_cost   = (q_prompt/1000.0)*INPUT_COST_PER_1K + (q_comp/1000.0)*OUTPUT_COST_PER_1K

            print(f"[Q{i+1}] prompt_tokens={q_prompt}  completion_tokens={q_comp}  cost=${q_cost:.6f}  time={q_time:.2f}s")
            print(f"Completed question {i+1}")

        # Set to append to existing file if not starting from Q1
        mode = 'a' if start_index>0 else 'w'
        print(f"Saving results to {output_file}...")
        with open(output_file, mode) as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        print(f"Results saved in {output_file}")

        # Calculate cost and time
        overall_prompt = self.handler.prompt_llm_token_count     - overall_prompt0
        overall_comp   = self.handler.completion_llm_token_count - overall_comp0
        overall_time   = time.time() - overall_t0
        overall_cost   = (overall_prompt/1000.0)*INPUT_COST_PER_1K + (overall_comp/1000.0)*OUTPUT_COST_PER_1K

        print("\n=== Run summary ===")
        print(f"Total prompt tokens:     {overall_prompt}")
        print(f"Total completion tokens: {overall_comp}")
        print(f"Total tokens:            {overall_prompt + overall_comp}")
        print(f"Estimated total cost:    ${overall_cost:.6f}")
        print(f"Total wall time:         {overall_time:.2f}s")

def create_agent(agent_class, callback_manager=None):
    # Build the LLM with token tracker
    llm = OpenAI(model="gpt-4.1-mini",
                 max_tokens=500,                 # Cap response length
                 callback_manager=callback_manager)
    agent_instance = agent_class(llm=llm)
    memory = agent_instance._create_memory()
    agent_final = agent_class(llm=llm, memory=memory)
    return agent_final

def main():
    
    # If starting from specific question in dataset
    start_index = 0
    num_questions = 10

    handler = TokenCountingHandler()
    cb_manager = CallbackManager([handler])

    runner = LongMemEvalRunner(SummaryAgent, cb_manager, handler)
    
    # Change according to dataset in use
    data_file = "asdrp/eval/data/custom_history/longmemeval_s.json"

    # Change according to agent analyzed
    output_file = "results/summary_agent_responses.json"

    os.makedirs("results", exist_ok=True)
    
    asyncio.run(runner.evaluate_on_dataset(data_file, output_file, start_index=start_index, num_questions=num_questions))
    
    """
    data_file = "asdrp/eval/data/custom_history/longmemeval_s.json"
    print("Hello World")
    """

if __name__ == "__main__":
    main()

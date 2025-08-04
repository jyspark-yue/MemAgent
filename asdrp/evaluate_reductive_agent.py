import json
import asyncio
from asdrp.agent.reductive_agent import ReductiveAgent

class LongMemEvalRunner:
    def __init__(self, agent):
        self.agent = create_agent(agent)
    
    async def process_chat_history(self, haystack_sessions, question):
        print(f"Processing {len(haystack_sessions)} haystack sessions...")
        
        session_count = 0
        for session in haystack_sessions:
            session_count += 1
            if session_count % 50 == 0:  # Print progress every 50 sessions
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
    
    def create_agent(agent_class):
        agent_instance = agent_class()
        memory = agent_instance._create_memory()
        agent_final = agent_class(memory=memory)

    async def evaluate_on_dataset(self, data_file, output_file, max_questions):
        print(f"Loading dataset from {data_file}...")
        with open(data_file, 'r') as f:
            dataset = json.load(f)
        
        print(f"Dataset loaded with {len(dataset)} total questions")
        
        if(max_questions):
            dataset = dataset[:max_questions]
            print(f"Processing first {len(dataset)} questions")
        
        results = []

        for i, item in enumerate(dataset):
            print(f"Processing question {i+1}/{len(dataset)}: {item.get('question_id', 'unknown')}")
            
            # Check the structure of the item
            print(f"Item keys: {list(item.keys())}")
            print(f"Number of haystack sessions: {len(item.get('haystack_sessions', []))}")
            
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
            
            print(f"Completed question {i+1}")

        print(f"Saving results to {output_file}...")
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"Results saved in {output_file}")


def create_agent(agent_class):
    agent_instance = agent_class()
    memory = agent_instance._create_memory()
    agent_final = agent_class(memory=memory)
    return agent_final

def main():
    """
    runner = LongMemEvalRunner(ReductiveAgent)
    
    data_file = "asdrp/eval/data/custom_history/longmemeval_m.json"
    output_file = "results/reductive_agent_results.json"
    max_questions = 1 
    

    import os
    os.makedirs("results", exist_ok=True)
    
    asyncio.run(runner.evaluate_on_dataset(data_file, output_file, max_questions))
    """
    data_file = "asdrp/eval/data/custom_history/longmemeval_m.json"
    print("Hello World")
    

if __name__ == "__main__":
    main()



    



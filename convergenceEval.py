import os
import json
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class ConvergenceEvaluator:
    def __init__(self):
        self.conversations = []

    def add_conversation(self, user_queries, bot_responses):
        """
        Add a conversation to evaluate.
        :param user_queries: List of user queries.
        :param bot_responses: Corresponding bot responses.
        """
        self.conversations.append({"queries": user_queries, "responses": bot_responses})

    def evaluate(self):
        """
        Evaluate all stored conversations for convergence.
        :return: List of evaluation results for each conversation.
        """
        results = []
        for convo in self.conversations:
            queries = convo["queries"]
            responses = convo["responses"]

            progression_score = self._evaluate_task_progression(responses)
            loop_score = self._evaluate_loops(responses)
            convergence_score = self._evaluate_convergence(responses)

            feedback = self._generate_feedback(progression_score, loop_score, convergence_score)

            results.append({
                "queries": queries,
                "responses": responses,
                "progression_score": progression_score,
                "loop_score": loop_score,
                "convergence_score": convergence_score,
                "feedback": feedback
            })
        return results

    def _evaluate_task_progression(self, responses):
        """
        Check if the responses become progressively more relevant and refined.
        :param responses: List of bot responses.
        :return: Score (0-1) for task progression.
        """
        progression_count = 0
        for i in range(1, len(responses)):
            if len(responses[i]) > len(responses[i - 1]) and responses[i] != responses[i - 1]:
                progression_count += 1
        return progression_count / max(1, len(responses) - 1)

    def _evaluate_loops(self, responses):
        """
        Check for repetitive responses (loops).
        :param responses: List of bot responses.
        :return: Score (0-1), where 0 = stuck in loops, 1 = no loops.
        """
        unique_responses = len(set(responses))
        return unique_responses / len(responses)

    def _evaluate_convergence(self, responses):
        """
        Check if the conversation converges to an outcome (meaningful resolution).
        :param responses: List of bot responses.
        :return: Score (0-1), where 1 = clear convergence.
        """
        return 1 if len(responses[-1].strip()) > 0 and "I don't know" not in responses[-1] else 0

    def _generate_feedback(self, progression_score, loop_score, convergence_score):
        """
        Generate feedback based on evaluation scores.
        :return: Feedback string.
        """
        if convergence_score == 1 and progression_score > 0.8:
            return "The agent is converging well and progressing effectively toward its goal."
        elif loop_score < 0.5:
            return "The agent appears to be stuck in repetitive responses. Consider improving task logic."
        else:
            return "The agent's responses show room for improvement in progressing toward a clear outcome."

    def export_results(self, results, filename="convergenceResult.json"):
        """
        Export evaluation results to a JSON file.
        :param results: List of evaluation results.
        :param filename: File name for the exported JSON file.
        """
        with open(filename, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Results exported to {filename}")

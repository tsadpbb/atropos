import openai
import random

class TeacherAgent:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.student_profile = {}
        self.weak_areas = []
        self.question_history = []

    def load_student_profile(self, profile_dict):
        self.student_profile = profile_dict
        self.weak_areas = []
        self.question_history = []

    def _build_prompt(self, topic, difficulty, goal):
        return (
            f"You are an expert teacher helping a student learn about '{topic}' at a '{difficulty}' level.\n"
            f"The student is learning for this goal: '{goal}'.\n"
            "Start with a clear, engaging multiple-choice question to assess their baseline understanding."
        )

    def generate_initial_question(self):
        topic = self.student_profile.get("topic")
        difficulty = self.student_profile.get("difficulty")
        goal = self.student_profile.get("goal")

        prompt = self._build_prompt(topic, difficulty, goal)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful, adaptive teacher."},
                {"role": "user", "content": prompt}
            ]
        )

        question = response['choices'][0]['message']['content']
        self.question_history.append({"question": question, "type": "initial"})
        return question

    def evaluate_student_response(self, question, student_answer):
        prompt = (
            f"You asked: {question}\n"
            f"The student answered: {student_answer}\n"
            "Evaluate the correctness. If incorrect, explain why in simple terms and identify the specific weak area."
        )

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert teacher scoring a student response."},
                {"role": "user", "content": prompt}
            ]
        )

        feedback = response['choices'][0]['message']['content']

        # Optional: extract weak area via LLM or regex (simplified here)
        if "incorrect" in feedback.lower() or "not correct" in feedback.lower():
            self.weak_areas.append("unidentified weak topic")

        self.question_history[-1]["student_answer"] = student_answer
        self.question_history[-1]["teacher_feedback"] = feedback

        return feedback

    def follow_up_on_weakness(self):
        if not self.weak_areas:
            return "Student has no identified weaknesses so far."

        weak_topic = self.weak_areas[-1]
        prompt = (
            f"You are helping a student understand '{self.student_profile['topic']}' better.\n"
            f"They are struggling with: '{weak_topic}'.\n"
            "Explain this weak area briefly and follow up with another multiple-choice question."
        )

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a tutor addressing student knowledge gaps."},
                {"role": "user", "content": prompt}
            ]
        )

        question = response['choices'][0]['message']['content']
        self.question_history.append({"question": question, "type": "followup", "weak_topic": weak_topic})
        return question

#!/usr/bin/env python3
"""
ExamCraft: Enhanced Adaptive Teacher - Trains students to mastery!

This version continues teaching until the student reaches high proficiency in ALL topics,
demonstrating persistent adaptive teaching behavior.
"""

import os
import json
import random
from typing import Dict, List, Any, Tuple, Optional

class EnhancedExamCraftDemo:
    """
    Enhanced ExamCraft that teaches until student reaches mastery in all topics.
    Perfect for demonstrating adaptive teaching to completion!
    """
    
    def __init__(self, profile_path: str = "example_profile.json"):
        # Load student profile
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as file:
                self.profile = json.load(file)
        else:
            self.profile = self._create_default_profile()
        
        # Mastery threshold - student needs to reach this in ALL topics
        self.mastery_threshold = 0.85  # 85% proficiency
        
        # Initialize student metrics
        self.reset_student()
        
        # Track teaching session
        self.question_count = 0
        self.session_history = []
        
    def _create_default_profile(self) -> Dict[str, Any]:
        """Create default student profile."""
        return {
            "student_id": "adaptive_learner_001",
            "target_grade": "11th grade",
            "learning_goal": "Master all linear algebra concepts",
            "topics": [
                {"name": "vectors", "proficiency": 0.45},
                {"name": "matrices", "proficiency": 0.30},
                {"name": "linear_systems", "proficiency": 0.25},
                {"name": "eigenvalues_eigenvectors", "proficiency": 0.20}
            ],
            "preferred_learning_style": "visual"
        }
    
    def reset_student(self):
        """Reset student to initial proficiency levels."""
        self.student_metrics = {}
        for topic in self.profile["topics"]:
            self.student_metrics[topic["name"]] = topic["proficiency"]
        
        self.question_count = 0
        self.session_history = []
    
    def check_mastery(self) -> Tuple[bool, Dict[str, float]]:
        """Check if student has reached mastery in all topics."""
        topic_status = {}
        all_mastered = True
        
        for topic, proficiency in self.student_metrics.items():
            is_mastered = proficiency >= self.mastery_threshold
            topic_status[topic] = proficiency
            if not is_mastered:
                all_mastered = False
        
        return all_mastered, topic_status
    
    def get_weakest_topic(self) -> str:
        """Find the topic where student is struggling most."""
        return min(self.student_metrics.items(), key=lambda x: x[1])[0]
    
    def simulate_teaching_session(self) -> Dict[str, Any]:
        """
        Simulate a complete teaching session until mastery.
        This is what you'll demo in your video!
        """
        print("🎓 ExamCraft: Teaching to Mastery Demo")
        print("=" * 50)
        
        # Show initial state
        print("\n📊 Initial Student Proficiency:")
        for topic, prof in self.student_metrics.items():
            print(f"  {topic}: {prof:.1%}")
        
        print(f"\n🎯 Goal: Reach {self.mastery_threshold:.0%} proficiency in ALL topics")
        print("🚀 Starting adaptive teaching session...\n")
        
        session_results = []
        round_num = 1
        
        # Keep teaching until mastery achieved
        while True:
            print(f"--- Round {round_num} ---")
            
            # Check current status
            all_mastered, topic_status = self.check_mastery()
            
            if all_mastered:
                print("\n🎉 CONGRATULATIONS! Student has achieved mastery in ALL topics!")
                break
            
            # Find weakest topic and generate question
            target_topic = self.get_weakest_topic()
            current_prof = self.student_metrics[target_topic]
            
            # Determine appropriate difficulty
            if current_prof < 0.4:
                difficulty = "easy"
            elif current_prof < 0.7:
                difficulty = "medium"
            else:
                difficulty = "hard"
            
            print(f"🎯 Targeting weakest topic: {target_topic} ({current_prof:.1%})")
            print(f"📝 Generating {difficulty} question...")
            
            # Simulate question and answer
            question_result = self._simulate_question_answer(target_topic, difficulty)
            
            # Update student learning
            self._update_student_learning(target_topic, question_result["is_correct"])
            
            # Track progress
            self.question_count += 1
            self.session_history.append(question_result)
            session_results.append(question_result)
            
            # Show immediate feedback
            result_icon = "✅" if question_result["is_correct"] else "❌"
            new_prof = self.student_metrics[target_topic]
            improvement = new_prof - current_prof
            
            print(f"{result_icon} Student answered: {'Correct' if question_result['is_correct'] else 'Incorrect'}")
            print(f"📈 {target_topic}: {current_prof:.1%} → {new_prof:.1%} (+{improvement:.1%})")
            
            # Show overall progress every 5 questions
            if self.question_count % 5 == 0:
                print("\n📊 Current Progress:")
                for topic, prof in self.student_metrics.items():
                    status = "✅ Mastered" if prof >= self.mastery_threshold else f"{prof:.1%}"
                    print(f"  {topic}: {status}")
                print()
            
            round_num += 1
            
            # Safety limit (prevent infinite loop in demo)
            if round_num > 30:
                print(f"\n⏰ Demo limit reached ({round_num-1} rounds)")
                break
        
        # Final summary
        print("\n" + "=" * 50)
        print("📋 TEACHING SESSION COMPLETE!")
        print(f"✨ Total questions asked: {self.question_count}")
        print(f"🎯 Rounds of teaching: {round_num - 1}")
        
        # Final proficiency
        print("\n🏆 Final Student Proficiency:")
        all_mastered, final_status = self.check_mastery()
        for topic, prof in final_status.items():
            status = "🎉 MASTERED" if prof >= self.mastery_threshold else f"{prof:.1%}"
            print(f"  {topic}: {status}")
        
        if all_mastered:
            print("\n🎊 MISSION ACCOMPLISHED: Student has mastered all topics!")
        else:
            print(f"\n📈 Progress made! Continue teaching to reach {self.mastery_threshold:.0%} in all topics.")
        
        return {
            "total_questions": self.question_count,
            "rounds": round_num - 1,
            "final_proficiencies": dict(self.student_metrics),
            "all_mastered": all_mastered,
            "session_history": session_results
        }
    
    def _simulate_question_answer(self, topic: str, difficulty: str) -> Dict[str, Any]:
        """Simulate a question and student's answer."""
        # Get student's current proficiency
        proficiency = self.student_metrics[topic]
        
        # Adjust success probability based on difficulty
        difficulty_modifiers = {
            "easy": 0.3,
            "medium": 0.0,
            "hard": -0.2
        }
        modifier = difficulty_modifiers.get(difficulty, 0.0)
        
        # Calculate success probability
        success_prob = max(0.15, min(0.95, proficiency + modifier))
        
        # Add momentum from recent performance
        if len(self.session_history) >= 2:
            recent_correct = sum(1 for item in self.session_history[-2:] 
                               if item["is_correct"])
            momentum = (recent_correct - 1) * 0.1
            success_prob = max(0.15, min(0.95, success_prob + momentum))
        
        # Determine if answer is correct
        is_correct = random.random() < success_prob
        
        # Sample questions for demo
        sample_questions = {
            "vectors": {
                "easy": "What is the result of adding two vectors?",
                "medium": "How do you calculate the dot product of two vectors?",
                "hard": "Prove that the dot product is commutative."
            },
            "matrices": {
                "easy": "What is a matrix?",
                "medium": "How do you multiply two matrices?",
                "hard": "When does a matrix inverse exist?"
            },
            "linear_systems": {
                "easy": "What is a system of linear equations?",
                "medium": "How does Gaussian elimination work?",
                "hard": "Analyze the conditions for system solvability."
            },
            "eigenvalues_eigenvectors": {
                "easy": "What is an eigenvalue?",
                "medium": "How do you find eigenvalues of a 2x2 matrix?",
                "hard": "Prove the relationship between trace and eigenvalues."
            }
        }
        
        question = sample_questions.get(topic, {}).get(difficulty, f"Sample {difficulty} question about {topic}")
        
        return {
            "topic": topic,
            "difficulty": difficulty,
            "question": question,
            "is_correct": is_correct,
            "success_probability": success_prob,
            "proficiency_before": proficiency
        }
    
    def _update_student_learning(self, topic: str, is_correct: bool):
        """Update student's proficiency based on learning."""
        current_prof = self.student_metrics[topic]
        
        if is_correct:
            # Successful answer - moderate improvement
            improvement = 0.05 + random.uniform(0, 0.03)
        else:
            # Failed answer but learned from explanation - smaller improvement
            improvement = 0.02 + random.uniform(0, 0.02)
        
        # Diminishing returns as proficiency increases
        diminishing_factor = (1.0 - current_prof) * 0.8 + 0.2
        improvement *= diminishing_factor
        
        # Update proficiency
        new_prof = min(1.0, current_prof + improvement)
        self.student_metrics[topic] = new_prof
    
    def run_quick_demo(self):
        """Run a quick 3-round demo for video purposes."""
        print("🎓 ExamCraft: Quick Demo - Teaching to Mastery")
        print("=" * 45)
        
        # Show initial state
        print("\n📊 Student Starting Proficiency:")
        for topic, prof in self.student_metrics.items():
            print(f"  {topic}: {prof:.1%}")
        
        # Set a lower threshold for demo
        original_threshold = self.mastery_threshold
        self.mastery_threshold = 0.65  # Lower for demo
        
        print(f"\n🎯 Demo Goal: Reach {self.mastery_threshold:.0%} in all topics")
        print("🚀 Let's see adaptive teaching in action!\n")
        
        for round_num in range(1, 4):  # Just 3 rounds for demo
            print(f"--- Round {round_num} ---")
            
            # Find and target weakest topic
            target_topic = self.get_weakest_topic()
            current_prof = self.student_metrics[target_topic]
            
            print(f"🎯 Focusing on: {target_topic} ({current_prof:.1%})")
            
            # Simulate teaching
            question_result = self._simulate_question_answer(target_topic, "medium")
            self._update_student_learning(target_topic, question_result["is_correct"])
            
            new_prof = self.student_metrics[target_topic]
            result_icon = "✅" if question_result["is_correct"] else "❌"
            
            print(f"{result_icon} Result: {new_prof:.1%} ({new_prof-current_prof:+.1%})")
            
            self.question_count += 1
        
        # Reset threshold
        self.mastery_threshold = original_threshold
        
        print("\n📈 Final Progress:")
        for topic, prof in self.student_metrics.items():
            print(f"  {topic}: {prof:.1%}")
        
        print("\n✨ This is how ExamCraft adapts teaching until mastery!")


def main():
    """Demo the enhanced ExamCraft environment."""
    demo = EnhancedExamCraftDemo()
    
    print("Choose demo type:")
    print("1. Quick demo (for video recording)")
    print("2. Full teaching session (until mastery)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        demo.run_quick_demo()
    else:
        demo.simulate_teaching_session()
    
    print("\n🎬 Perfect for demo video! Shows adaptive teaching to mastery.")


if __name__ == "__main__":
    main()
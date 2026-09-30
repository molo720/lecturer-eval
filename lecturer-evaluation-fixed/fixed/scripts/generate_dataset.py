import random
import pandas as pd
random.seed(42)
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)


clarity_positive = [
    "Explains clearly", "Makes topics easier to understand", "Breaks complex structures into understandable ones",
    "Simplifies difficult concepts well", "Uses good examples when teaching",
    "The way he breaks down ideas makes sense", "Every explanation is easy to follow",
    "I always leave the class understanding the topic", "Explains difficult material well"
]
clarity_negative = [
    "doesn't explain well", "Makes topics difficult to understand", "Complicates simple topics",
    "Rushes through explanations", "Uses confusing terms without breaking them down",
    "I struggle to follow most explanations", "Skips important details when explaining",
    "Lectures feel disorganized and hard to follow", "Hard to follow his explanations"
]
subject_knowledge_positive = [
    "Has deep knowledge of the subject", "Clearly knows the material inside and out",
    "Answers even the toughest questions confidently", "You can tell he's mastered this field",
    "Backs up explanations with real-world examples from experience", "Never seems caught off guard by a question",
    "I trust his understanding of even advanced topics", "Goes beyond the textbook when explaining concepts",
    "Explains difficult concepts with real depth"
]
subject_knowledge_negative = [
    "doesn't seem to know the material well", "Struggles to answer basic questions",
    "Gives vague answers when pushed for detail", "Seems to be reading straight from slides",
    "I've caught factual errors in lectures before", "Can't connect concepts to real applications",
    "Avoids questions that go beyond the syllabus", "His grasp of the subject feels shaky"
]

engagement_positive = [
    "Makes the class genuinely interesting", "Keeps everyone engaged the whole session",
    "Uses humor to keep the room awake", "Encourages students to participate often",
    "I actually look forward to this class", "Turns dry topics into lively discussions",
    "Involves the class with questions and debates", "His energy makes the lecture fly by"
]
engagement_negative = [
    "the class feels boring and flat", "Just reads off slides with no energy",
    "Rarely interacts with students during lectures", "People are on their phones the whole time",
    "I find it hard to stay awake in this class", "Never asks questions or invites discussion",
    "The lecture drags on with no variation", "Feels like a one-way monologue every time",
    "Hard to stay interested in this class"
]

assessment_positive = [
    "Grades fairly and consistently", "Gives detailed feedback on assignments",
    "Exams match what was actually taught", "Returns graded work quickly",
    "I always understand why I lost points", "Rubrics are clear before submitting work",
    "Feedback actually helps me improve next time", "Grading feels transparent and well-justified"
]
assessment_negative = [
    "grades inconsistently between students", "Feedback is vague or nonexistent",
    "Exams cover things never taught in class", "Takes weeks to return graded assignments",
    "I never know why I lost points", "No rubric is given before assignments are due",
    "Grading seems arbitrary from one submission to the next", "Comments on assignments are unhelpful or missing",
    "Help with grading disputes is hard to get"
]

availability_positive = [
    "Always available during office hours", "Responds to emails quickly",
    "Happy to answer questions after class", "Makes time for students who are struggling",
    "I've never had trouble reaching him for help", "Sets up extra sessions when needed",
    "Very approachable outside of lecture time", "Replies on the discussion forum within a day"
]
availability_negative = [
    "is nearly impossible to reach outside class", "Emails go unanswered for over a week",
    "Never holds actual office hours", "Seems annoyed when students ask for extra help",
    "I couldn't get a response before the deadline", "Cancels office hours without notice",
    "Hard to schedule any one-on-one time", "Ignores questions posted on the course forum",
    "Help is hard to get outside class"
]
course_name = ["Special Computing","Data Science","Cloud Computing","Machine Learning","Introduction to Computing"]
course_code = ["CSC410","CSC412","CSC406","CSC101","CSC408"]
lecturer_names = ["Dr Okafor","Dr Adeyemi","Dr Martins", "Dr Faith","Dr Pomele"]

aspects = {
    "Clarity of Instruction": {"positive": clarity_positive, "negative": clarity_negative},
    "Subject Knowledge": {"positive": subject_knowledge_positive, "negative": subject_knowledge_negative},
    "Engagement": {"positive": engagement_positive, "negative": engagement_negative},
    "Assessment": {"positive": assessment_positive, "negative": assessment_negative},
    "Availability": {"positive": availability_positive, "negative": availability_negative}
}
    
def generate_comment():
    num_aspects = random.randint(1, 3)
    chosen_aspects = random.sample(list(aspects.keys()), num_aspects)

    sentence_parts = []
    labels = []

    for aspect in chosen_aspects:
        sentiment = random.choice(["positive", "negative"])
        feedback = random.choice(aspects[aspect][sentiment])
        sentence_parts.append(feedback)
        labels.append((aspect, sentiment))

    if len(sentence_parts) == 1:
        comment_text = sentence_parts[0] + "."
    else:
        comment_text = sentence_parts[0]
        for i in range(1, len(sentence_parts)):
            part = sentence_parts[i]
            prev_sentiment = labels[i-1][1]
            curr_sentiment = labels[i][1]

            if prev_sentiment == curr_sentiment:
                connector = random.choice(["and", "also"])
            else:
                connector = random.choice(["but", "however", "although"])

            comment_text += f", {connector} {part.lower()}"
        comment_text += "."

    return comment_text, labels


def generate_rating(labels):
    positive_count = sum(1 for aspect, sentiment in labels if sentiment == "positive")
    negative_count = sum(1 for aspect, sentiment in labels if sentiment == "negative")

    if positive_count > negative_count:
        base_rating = random.choice([4,5])
    elif negative_count > positive_count:
        base_rating = random.choice([1,2])
    else:
        base_rating = 3

    if random.random()< 0.15:
        base_rating = random.randint(1,5)
    return base_rating

def generate_dataset(n=1000):
    data = []
    
    for i in range(n):
        comment, labels = generate_comment()
        rating = generate_rating(labels)
        course = random.choice(course_name)
        code = random.choice(course_code)
        lecturer = random.choice(lecturer_names)
        
        data.append({
            "course": course,
            "course_code": code,
            "lecturer_name": lecturer,
            "rating": rating,
            "comment": comment,
            "labels": labels
        })
    
    df = pd.DataFrame(data)
    return df


dataset = generate_dataset(1000)
dataset.to_csv("data/synthetic_evaluations.csv", index=False)
print(dataset.head())
print(f"Generated {len(dataset)} rows")


def generate_comment_with_bio():
    num_aspects = random.randint(1, 3)
    chosen_aspects = random.sample(list(aspects.keys()), num_aspects)

    sentence_parts = []
    labels = []

    for aspect in chosen_aspects:
        sentiment = random.choice(["positive", "negative"])
        feedback = random.choice(aspects[aspect][sentiment])
        sentence_parts.append(feedback)
        labels.append((aspect, sentiment))

    if len(sentence_parts) == 1:
        comment_text = sentence_parts[0] + "."
    else:
        comment_text = sentence_parts[0]
        for i in range(1, len(sentence_parts)):
            part = sentence_parts[i]
            prev_sentiment = labels[i-1][1]
            curr_sentiment = labels[i][1]
            if prev_sentiment == curr_sentiment:
                connector = random.choice(["and", "also"])
            else:
                connector = random.choice(["but", "however", "although"])
            comment_text += f", {connector} {part.lower()}"
        comment_text += "."

    tokens = word_tokenize(comment_text.lower())
    bio_tags = ["O"] * len(tokens)

    for aspect, sentiment in labels:
        phrase = None
        for s in ["positive", "negative"]:
            for p in aspects[aspect][s]:
                if p.lower() in comment_text.lower():
                    phrase = p.lower()
                    break
            if phrase:
                break
        
        if not phrase:
            continue

        phrase_tokens = word_tokenize(phrase)
        aspect_tag = aspect.replace(" ", "_").upper()

        for i in range(len(tokens) - len(phrase_tokens) + 1):
            if tokens[i:i + len(phrase_tokens)] == phrase_tokens:
                bio_tags[i] = f"B-{aspect_tag}"
                for j in range(1, len(phrase_tokens)):
                    bio_tags[i + j] = f"I-{aspect_tag}"
                break

    return comment_text, labels, tokens, bio_tags



def generate_dataset_with_bio(n=1000):
    data = []
    for i in range(n):
        comment, labels, tokens, bio_tags = generate_comment_with_bio()
        rating = generate_rating(labels)
        course = random.choice(course_name)
        code = random.choice(course_code)
        lecturer = random.choice(lecturer_names)

        data.append({
            "course": course,
            "course_code": code,
            "lecturer_name": lecturer,
            "rating": rating,
            "comment": comment,
            "labels": labels,
            "tokens": tokens,
            "bio_tags": bio_tags
        })

    df = pd.DataFrame(data)
    return df


bio_dataset = generate_dataset_with_bio(1000)
bio_dataset.to_csv("data/crf_training_data.csv", index=False)
print(bio_dataset[["tokens", "bio_tags"]].head(2))

print("\nClass distribution per aspect:")
for aspect in aspects.keys():
    pos = sum(1 for _, row in bio_dataset.iterrows() for a, s in row["labels"] if a == aspect and s == "positive")
    neg = sum(1 for _, row in bio_dataset.iterrows() for a, s in row["labels"] if a == aspect and s == "negative")
    print(f"  {aspect}: {pos} positive, {neg} negative")
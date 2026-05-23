from collections import defaultdict
from statistics import mean

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

from reportlab.platypus.flowables import Preformatted
from reportlab.platypus import Image

# import matplotlib.pyplot as plt


# def create_score_chart(scores):
#     plt.figure(figsize=(6, 3))
#     plt.plot(
#         range(1, len(scores)+1),
#         scores,
#         marker="o"
#     )
#     plt.xlabel("Question Number")
#     plt.ylabel("Score")
#     plt.title("Score Progression")
#     chart_path = "score_progression.png"
#     plt.tight_layout()
#     plt.savefig(chart_path)
#     plt.close()
#     return chart_path


# def create_topic_chart(topic_scores):

#     topics = list(topic_scores.keys())
#     averages = [
#         sum(v)/len(v)
#         for v in topic_scores.values()
#     ]
#     plt.figure(figsize=(7, 4))

#     plt.bar(
#         topics,
#         averages
#     )
#     plt.xticks(rotation=20)
#     plt.ylabel("Average Score")
#     plt.title("Topic-wise Performance")
#     chart_path = "topic_scores.png"
#     plt.tight_layout()
#     plt.savefig(chart_path)
#     plt.close()
#     return chart_path


# =========================================================
# Generate Interview Report PDF
# =========================================================

def generate_interview_report(
    interview_state: dict,
    output_path: str = "interview_report.pdf"
):

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    # =====================================================
    # Header
    # =====================================================

    elements.append(
        Paragraph(
            "AI Interview Evaluation Report",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1, 12)
    )

    # =====================================================
    # Candidate Info
    # =====================================================

    candidate_table = [

        ["Role", interview_state["job_role"]],

        ["Matched RAG Role",
         interview_state.get(
             "matched_role",
             "N/A"
         )],

        ["Experience",
         interview_state.get(
             "years_of_experience",
             "N/A"
         )],

        ["RAG Mode",
         interview_state.get(
             "rag_mode",
             "normal"
         )]
    ]

    table = Table(
        candidate_table,
        colWidths=[180, 320]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.lightgrey),

            ("GRID", (0, 0), (-1, -1),
             1, colors.black),

            ("FONTNAME", (0, 0), (-1, -1),
             "Helvetica")
        ])
    )

    elements.append(table)

    elements.append(
        Spacer(1, 20)
    )

    # =====================================================
    # Skills
    # =====================================================

    elements.append(
        Paragraph(
            "Extracted Skills",
            styles["Heading2"]
        )
    )

    skills_text = ", ".join(
        interview_state.get(
            "extracted_skills",
            []
        )
    )

    elements.append(
        Paragraph(
            skills_text,
            styles["BodyText"]
        )
    )

    elements.append(
        Spacer(1, 16)
    )

    # =====================================================
    # Overall Metrics
    # =====================================================

    evaluations = interview_state.get(
        "evaluations",
        []
    )

    scores = [
        e.get("score", 0)
        for e in evaluations
    ]

    avg_score = (
        round(mean(scores), 2)
        if scores else 0
    )

    correctness_counter = defaultdict(int)

    for e in evaluations:

        correctness_counter[
            e.get(
                "correctness",
                "Unknown"
            )
        ] += 1

    summary_data = [

        ["Metric", "Value"],

        ["Questions Asked",
         str(len(evaluations))],

        ["Average Score",
         str(avg_score)],

        ["Correct Answers",
         str(correctness_counter["Correct"])],

        ["Partially Correct",
         str(correctness_counter[
             "Partially Correct"
         ])],

        ["Incorrect",
         str(correctness_counter[
             "Incorrect"
         ])]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[250, 250]
    )

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.lightgrey),

            ("GRID", (0, 0), (-1, -1),
             1, colors.black)
        ])
    )

    elements.append(
        Paragraph(
            "Performance Summary",
            styles["Heading2"]
        )
    )

    elements.append(summary_table)

    elements.append(
        Spacer(1, 20)
    )


    topic_scores = defaultdict(list)

    for q, e in zip(
        interview_state["question_history"],
        evaluations
    ):

        topic_scores[
            q.get(
                "topic",
                "Unknown"
            )
        ].append(
            e.get("score", 0)
        )

    # score_chart = create_score_chart(scores)
    # topic_chart = create_topic_chart(
    #     topic_scores
    # )

    # elements.append(
    #     Paragraph(
    #         "Visual Analytics",
    #         styles["Heading2"]
    #     )
    # )

    # elements.append(
    #     Spacer(1, 12)
    # )

    # elements.append(
    #     Image(
    #         score_chart,
    #         width=420,
    #         height=220
    #     )
    # )

    # elements.append(
    #     Spacer(1, 20)
    # )

    # elements.append(
    #     Image(
    #         topic_chart,
    #         width=420,
    #         height=240
    #     )
    # )

    # elements.append(
    #     Spacer(1, 20)
    # )

    # elements.append(
    #     PageBreak()
    # )

    # =====================================================
    # Topic Breakdown
    # =====================================================

    topic_groups = defaultdict(list)

    for q, e in zip(
        interview_state["question_history"],
        evaluations
    ):

        topic_groups[q["topic"]].append((q, e))

    for topic, qa_pairs in topic_groups.items():

        elements.append(
            Paragraph(
                f"Topic: {topic}",
                styles["Heading1"]
            )
        )

        # =========================================
        # Topic-Level Retrieved Sources
        # =========================================

        first_q = qa_pairs[0][0]

        retrieved = first_q.get(
            "retrieved_context",
            []
        )

        if retrieved:

            elements.append(
                Paragraph(
                    "Retrieved Sources",
                    styles["Heading2"]
                )
            )

            for chunk in retrieved:

                compact_chunk = f"""
    <b>{chunk.get('source_book')}</b>
    (Page {chunk.get('page')},
    Similarity {round(chunk.get('score', 0), 2)})

    <br/><br/>

    {chunk.get('text')[:450]}...
    """

                elements.append(
                    Paragraph(
                        compact_chunk,
                        styles["BodyText"]
                    )
                )

                elements.append(
                    Spacer(1, 8)
                )

        elements.append(
            Spacer(1, 16)
        )

        # =========================================
        # Questions Under Topic
        # =========================================

        for idx, (q, e) in enumerate(
            qa_pairs,
            start=1
        ):

            qa_text = f"""
    <b>Question {idx}</b><br/><br/>

    <b>Difficulty:</b>
    {q.get('difficulty')}<br/><br/>

    <b>Subtopic:</b>
    {q.get('subtopic')}<br/><br/>

    <b>Question:</b><br/>
    {q.get('question')}<br/><br/>

    <b>Candidate Answer:</b><br/>
    {e.get('answer')}<br/><br/>

    <b>Score:</b>
    {e.get('score')} / 10<br/><br/>

    <b>Correctness:</b>
    {e.get('correctness')}<br/><br/>

    <b>Missing Points:</b><br/>
    {', '.join(e.get('missing_points', []))}<br/><br/>

    <b>Improvement Suggestion:</b><br/>
    {e.get('improvement_suggestion')}<br/><br/>

    <b>Reasoning:</b><br/>
    {q.get('reasoning')}
    """

            elements.append(
                Paragraph(
                    qa_text,
                    styles["BodyText"]
                )
            )

            elements.append(
                Spacer(1, 18)
            )

        elements.append(
            PageBreak()
        )

    # =====================================================
    # Finalize PDF
    # =====================================================

    doc.build(elements)

    return output_path



if __name__ == "__main__":
    import json

    # =========================================================
    # Load MongoDB State Dump
    # =========================================================

    with open(
        "sample_data.json",
        "r",
        encoding="utf-8"
    ) as f:

        mongo_doc = json.load(f)

    state = mongo_doc["state"]


    # =========================================================
    # Generate PDF
    # =========================================================

    output_path = generate_interview_report(
        state,
        "test_interview_report.pdf"
    )

    print(
        f"\nPDF generated successfully:\n{output_path}"
    )
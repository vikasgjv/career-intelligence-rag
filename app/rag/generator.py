import ollama


def generate_answer(results, question):

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
Evidence {i}:
Source: {result['source']}
Section: {result['section']}
Text: {result['text']}
"""
        )

    context = "\n".join(context_parts)

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": """
You are a career intelligence assistant.

Answer the question using ONLY the provided evidence.

Do not invent information.
Do not provide evidence or citations.
Return only a concise answer to the question.

If the evidence does not support the answer, say:
"The available evidence does not confirm this."
"""
            },
            {
                "role": "user",
                "content": f"""
Evidence:

{context}

Question:
{question}
"""
            }
        ]
    )

    return response["message"]["content"]
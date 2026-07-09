from groq import Groq
import os

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

def generate_medical_report(prediction, confidence):
    prompt = f"""
    You are a radiology assistant.
    Given CNN prediction: {prediction} and confidence: {confidence:.2f},
    generate a clinical style MRI report with Findings, Impression, Recommendations.
    Keep it professional and concise.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional radiologist assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

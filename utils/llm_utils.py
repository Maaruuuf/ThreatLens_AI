from groq import Groq


def build_final_prompt(incident_data, historical_context, mitigation_context):

    confidence = incident_data["confidence"]

    
    if confidence < 0.5:
        confidence_instruction = """
CONFIDENCE LEVEL: LOW

The ML prediction is uncertain and may be incorrect.

You MUST:
• Critically evaluate the predicted attack.
• Actively look for conflicting evidence in retrieved cases.
• Consider alternative attack types if supported by evidence.
"""
    elif confidence < 0.8:
        confidence_instruction = """
CONFIDENCE LEVEL: MODERATE

The ML prediction may be partially correct.

You SHOULD:
• Validate consistency with retrieved cases.
• Highlight any conflicting patterns.
• Consider alternative explanations if necessary.
"""
    else:
        confidence_instruction = """
CONFIDENCE LEVEL: HIGH

The ML prediction is likely reliable.

You SHOULD:
• Validate consistency with retrieved cases.
• Only challenge the prediction if strong contradictory evidence exists.
"""

    system_prompt = """
You are a senior cybersecurity SOC analyst specializing in network intrusion detection and incident analysis.

Your task is to analyze a network traffic incident using the following sources of evidence:

1. Machine learning prediction output
2. Retrieved historical network traffic cases
3. Retrieved mitigation knowledge mapped to security frameworks (MITRE ATT&CK and NIST)
4. Provide DETAILED mitigation recommendations using the provided mitigation knowledge.

IMPORTANT:
- Do NOT summarize into short bullet points only.
- Expand each mitigation with explanation.
- Use the provided mitigation descriptions to explain HOW and WHY the mitigation works.
- Reference mitigation IDs (e.g., M1031, RA-5) in your explanation.
- Explain the practical implementation of each mitigation in a real network environment.


STRICT ANALYSIS RULES

• Only use the information provided in the incident packet and retrieved evidence.
• Never invent network traffic features or behaviors that are not present.
• If evidence for a feature is missing, explicitly state: "No evidence available".
• Prioritize historical cases with higher similarity or rerank scores.
• Clearly distinguish between:
  - Confirmed indicators
  - Weak indicators
  - Missing or uncertain evidence.
• Treat the ML prediction as a hypothesis that must be validated using retrieved evidence.
• You are allowed to challenge or override the ML prediction ONLY if strong evidence supports an alternative attack type.
• The final response must be written as a professional SOC analyst report.

Your reasoning should focus on technical traffic behavior patterns and attack characteristics.
"""

    user_prompt = f"""
================ INCIDENT ALERT ================

Machine Learning Predicted Attack:
{incident_data['predicted_label']}

Prediction Confidence:
{confidence:.4f}

{confidence_instruction}

Observed Network Traffic Behavior:
{incident_data['incident_packet']}


================ RETRIEVED HISTORICAL CASES ================

These cases were retrieved using semantic similarity search and reranking.

{historical_context}


================ RETRIEVED MITIGATION KNOWLEDGE ================

{mitigation_context}


================ ANALYSIS TASK ================

Perform a structured security analysis of the incident using the retrieved evidence.

Step 1 — Explain the predicted attack type
Describe what the predicted attack typically represents in real network environments.

Step 2 — Evidence comparison
Compare the observed traffic behavior with patterns from the retrieved historical cases.

Step 3 — Feature-level analysis
Perform a structured comparison of traffic characteristics.

Step 4 — Indicators assessment
Identify:

• Strong indicators (clear matches with historical attack patterns)
• Weak indicators (partial matches)
• Missing or uncertain evidence

Step 5 — ML prediction validation and override analysis
• Evaluate whether the ML prediction is correct based on evidence.
• If conflicting evidence exists, clearly explain why.
• If a different attack type is more consistent, propose the alternative and justify it.

Step 6 — Defensive recommendations
Recommend mitigation actions based on the retrieved mitigation knowledge.

Step 7 — Risk evaluation
Estimate the severity of the incident.

Use this severity scale:

Low → reconnaissance or low-risk activity  
Medium → suspicious activity with moderate risk  
High → confirmed attack indicators  
Critical → active attack causing significant impact  


================ OUTPUT FORMAT ================

Produce the response strictly using this SOC report structure:

SOC INCIDENT REPORT

Predicted Attack:

Confidence Assessment:

Final Attack Assessment:
(Confirm or override the ML prediction with justification)

Feature Comparison Table:
Feature | Incident Behavior | Historical Pattern | Match Strength

Strong Indicators:
Weak or Uncertain Indicators:
Missing Evidence:

Relevant Historical Cases:
(Explain which cases are most similar and why)

Recommended Mitigations:
1. [Mitigation Name] (ID)
   - Explanation:
   - Why it works:
   - How to implement:

2. ...

Severity Assessment:

Analyst Notes:
(Provide concise expert reasoning)
"""

    return system_prompt, user_prompt


def run_llm(groq_api_key, system_prompt, user_prompt):
    client = Groq(api_key=groq_api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        max_tokens=1600
    )

    return response.choices[0].message.content 
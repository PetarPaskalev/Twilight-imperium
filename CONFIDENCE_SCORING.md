# Confidence Scoring Implementation

## Overview
The Twilight Imperium chatbot now includes confidence scoring for all responses. This feature uses OpenAI's **log probabilities (logprobs)** to calculate how confident the model is in its response.

## How It Works

### Technical Implementation
1. **Log Probabilities**: The OpenAI API returns log probabilities for each token generated
2. **Conversion**: We convert log probabilities to actual probabilities using `exp(logprob)`
3. **Averaging**: We calculate the average probability across all tokens in the response
4. **Score Range**: Final score is between 0.0 and 1.0

### Score Interpretation
- **0.9 - 1.0**: Very high confidence (model is very certain about the response)
- **0.7 - 0.9**: High confidence (model is confident, typical for clear factual answers)
- **0.5 - 0.7**: Moderate confidence (model has some uncertainty)
- **0.3 - 0.5**: Low confidence (model is uncertain, response may be speculative)
- **Below 0.3**: Very low confidence (model is very uncertain)

### Where to See It
Confidence scores are **logged in the backend only** and will NOT be shown to users. Look for log entries like:

```
📊 [CONFIDENCE SCORE] 0.8234 (0.0-1.0 scale)
```

## Backend Logs
When running the FastAPI backend (`twilight_api.py`), you'll see confidence scores in the terminal/console output for each response generated.

Example log output:
```
INFO:     127.0.0.1:52840 - "POST /chat HTTP/1.1" 200 OK
📊 [CONFIDENCE SCORE] 0.8567 (0.0-1.0 scale)
```

## Use Cases
- **Monitoring**: Identify responses where the model is uncertain
- **Quality Control**: Flag low-confidence responses for human review
- **Analytics**: Track average confidence scores over time
- **Debugging**: Correlate confidence with user satisfaction or accuracy

## Notes
- Confidence scores are **not** returned to the frontend
- They are **only visible in backend logs**
- The feature uses OpenAI's native logprobs functionality
- No additional API calls or costs beyond the standard OpenAI API usage


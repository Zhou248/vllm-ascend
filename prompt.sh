curl http://localhost:10037/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3",
    "prompt": [
      "San Francisco is",
      "Hello, who are you?",
      "Introduce yourself in one sentence.",
      "What is the capital of France?",
      "Write a short poem about the moon.",
      "Explain what machine learning is in simple words.",
      "What is 2 + 2 * 3?",
      "Tell me a joke about programmers.",
      "Summarize the importance of cybersecurity in one paragraph.",
      "Give me three tips for learning Python.",
      "Translate '\''Good morning'\'' into Chinese."
    ],
    "max_tokens": 164,
    "temperature": 0
  }'
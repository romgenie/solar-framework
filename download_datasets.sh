#\!/bin/bash

# Script to download benchmark datasets for SOLAR

echo "Creating dataset directories..."
mkdir -p datasets/{gsm8k,aqua_rat,logiqa,crt}

echo "Downloading GSM8K dataset..."
# Grade School Math dataset
curl -L https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl -o datasets/gsm8k/train.jsonl
curl -L https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o datasets/gsm8k/test.jsonl

echo "Downloading AQUA-RAT dataset..."
# AQUA dataset
curl -L https://raw.githubusercontent.com/deepmind/AQuA/master/data/AQuA.json -o datasets/aqua_rat/aqua.json

echo "Downloading LogiQA dataset..."
# LogiQA dataset
curl -L https://github.com/lgw863/LogiQA-dataset/raw/master/Train.txt -o datasets/logiqa/train.txt
curl -L https://github.com/lgw863/LogiQA-dataset/raw/master/Eval.txt -o datasets/logiqa/eval.txt
curl -L https://github.com/lgw863/LogiQA-dataset/raw/master/Test.txt -o datasets/logiqa/test.txt

echo "Creating CRT dataset..."
# CRT - Creating sample file as there is no official repository
cat > datasets/crt/crt_problems.json << EOL
[
  {
    "id": 1,
    "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "intuitive_answer": "$0.10",
    "correct_answer": "$0.05",
    "explanation": "Let x be the cost of the ball. Then the bat costs x + $1.00. We have x + (x + $1.00) = $1.10, so 2x + $1.00 = $1.10, thus 2x = $0.10, and x = $0.05."
  },
  {
    "id": 2,
    "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "intuitive_answer": "100 minutes",
    "correct_answer": "5 minutes",
    "explanation": "Each machine makes 1 widget in 5 minutes. So 100 machines would make 100 widgets in 5 minutes."
  },
  {
    "id": 3,
    "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
    "intuitive_answer": "24 days",
    "correct_answer": "47 days",
    "explanation": "If the lake is fully covered on day 48, and the patch doubles each day, then the lake was half covered on day 47."
  }
]
EOL

echo "All datasets downloaded successfully\!"
echo "Dataset structure:"
ls -R datasets/

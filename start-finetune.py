"""
Fine-tune a chat model using OpenAI Python client (v1+).

What this script does:
1) Uploads qa_dataset.jsonl as a training file (purpose="fine-tune")
2) Starts a fine-tune job on BASE_MODEL
3) Polls status until the job finishes (succeeded/failed)
4) When succeeded, prints the fine-tuned model id and demonstrates one sample call.

Requirements:
- openai>=1.0 (the 'OpenAI' client class)
- python-dotenv (optional) if you want to load .env
- qa_dataset.jsonl in the same folder (chat-style entries)
"""

import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # optional: loads OPENAI_API_KEY from .env

# ----------------------------
# Configuration
# ----------------------------
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

client = OpenAI(api_key=API_KEY)

TRAINING_FILE_PATH = "qa_big_dataset.jsonl"   # path to the file you created earlier
BASE_MODEL = "gpt-4.1-nano-2025-04-14"              # replace with a model your account can fine-tune
POLL_INTERVAL = 30                        # seconds between polling job status

# ----------------------------
# Helper: upload file for fine-tuning
# ----------------------------
def upload_training_file(path):
    print(f"Uploading {path} ...")
    with open(path, "rb") as f:
        # purpose should be "fine-tune" per the docs
        res = client.files.create(file=f, purpose="fine-tune")
    # response contains an id like "file-xxxx"
    file_id = getattr(res, "id", None) or res.get("id")
    print(f"Uploaded. file_id={file_id}")
    return file_id

# ----------------------------
# Helper: start fine-tune job
# ----------------------------
def create_fine_tune_job(training_file_id, base_model):
    print(f"Creating fine-tune job (base_model={base_model}) ...")
    job = client.fine_tuning.jobs.create(training_file=training_file_id, 
                                        model=base_model,
                                        hyperparameters={
        "n_epochs": 3,
        "batch_size": 8,
    })
    job_id = getattr(job, "id", None) or job.get("id")
    print(f"Started fine-tune job: {job_id}")
    return job_id

# ----------------------------
# Helper: poll fine-tune job until complete
# ----------------------------
def wait_for_job(job_id, poll_interval=30):
    print(f"Polling job {job_id} every {poll_interval}s ...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = getattr(job, "status", None) or job.get("status")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] status={status}")
        if status in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(poll_interval)

# ----------------------------
# Helper: call the fine-tuned chat model
# ----------------------------
def call_fine_tuned_model(model_id, user_prompt):
    # Chat-style call
    messages = [
        {"role": "system", "content": "You are an expert in economics. Answer concisely and only use what's in your training data."},
        {"role": "user", "content": user_prompt}
    ]
    resp = client.chat.completions.create(model=model_id, messages=messages, max_tokens=400)
    # Extract the assistant message
    answer = resp.choices[0].message.content
    return answer

# ----------------------------
# Main flow
# ----------------------------
def main():
    # 1) Upload file
    if not os.path.exists(TRAINING_FILE_PATH):
        raise FileNotFoundError(f"Training file not found: {TRAINING_FILE_PATH}")

    training_file_id = upload_training_file(TRAINING_FILE_PATH)

    # 2) Create fine-tune job
    job_id = create_fine_tune_job(training_file_id, BASE_MODEL)

    # 3) Poll / wait
    job_result = wait_for_job(job_id, poll_interval=POLL_INTERVAL)

    # 4) Inspect result and get fine_tuned_model id if succeeded
    status = getattr(job_result, "status", None) or job_result.get("status")
    if status == "succeeded":
        # Depending on API response shape, the fine-tuned model id may be:
        # - job_result.fine_tuned_model OR job_result.result?.fine_tuned_model
        fine_tuned_model = getattr(job_result, "fine_tuned_model", None) or job_result.get("fine_tuned_model")
        # Some older responses include 'result' or 'fine_tuned_model' in different places; fallback to scanning:
        if not fine_tuned_model:
            # Try scanning top-level fields
            for k in ("result", "fine_tuned_model", "fine_tuned_models"):
                val = getattr(job_result, k, None) or job_result.get(k)
                if isinstance(val, str) and val.startswith("ft:"):
                    fine_tuned_model = val
                    break
                if isinstance(val, list) and val:
                    fine_tuned_model = val[0]
                    break

        print("Fine-tune succeeded!")
        print("Job result (raw):")
        try:
            print(json.dumps(job_result, default=lambda o: getattr(o, "__dict__", str(o)), indent=2))
        except Exception:

            print(job_result)

        if fine_tuned_model:
            print(f"Fine-tuned model id: {fine_tuned_model}")
            # 5) Example interaction
            sample_question = "How does the extent of the market limit the division of labor?"
            print(f"\nCalling fine-tuned model '{fine_tuned_model}' with sample prompt...")
            answer = call_fine_tuned_model(fine_tuned_model, sample_question)
            print("=== MODEL RESPONSE ===")
            print(answer)
        else:
            print("Could not find fine_tuned_model id in job result. Inspect job_result to locate it.")
    else:
        print(f"Fine-tune job did not succeed. status={status}")
        print("Job result:")
        print(job_result)

if __name__ == "__main__":
    main()

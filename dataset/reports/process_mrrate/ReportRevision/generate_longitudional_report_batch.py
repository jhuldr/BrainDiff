import sys
sys.path.insert(0, "/path/to/code/BrainDiff")

import io
import json
import time
import pandas as pd
from pathlib import Path

from .generate_longitudional_report import LongitudinalReportGenerator


class LongitudinalReportBatchGenerator(LongitudinalReportGenerator):
    """
    Same longitudinal generation as the synchronous version, but routed through
    OpenAI's Batch API. Each request carries one chunk of cases; the arbitrary
    per-row index inside the prompt keeps outputs aligned on parse.
    """

    def create_batch_file(self, meta_df: pd.DataFrame, jsonl_path, batching: int = 8) -> int:
        count = 0
        with open(jsonl_path, "w") as f:
            for chunk in self.iter_chunks(meta_df, chunk_size=batching):
                system_prompt, user_prompt = self._build_prompt(chunk)
                request = {
                    "custom_id": f"chunk-{chunk.index[0]}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.model,
                        "input": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "reasoning": {"effort": "low"},
                        "prompt_cache_key": "longitudinal-generation",
                    },
                }
                f.write(json.dumps(request) + "\n")
                count += 1
        print(f"[INFO] Wrote {count} batch requests to {jsonl_path}")
        return count

    def submit_batch(self, jsonl_path) -> str:
        batch_file = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        print(f"[INFO] Submitted batch {batch.id}")
        return batch.id

    def poll_batch(self, batch_id: str, interval: int = 30):
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            batch = self.client.batches.retrieve(batch_id)
            print(f"[INFO] Batch {batch_id} status: {batch.status}")
            if batch.status in terminal:
                return batch
            time.sleep(interval)

    def retrieve_and_parse(self, meta_df, batch, output_path, out_name) -> pd.DataFrame:
        if batch.status != "completed":
            print(f"[ERROR] Batch ended with status {batch.status}. Nothing to parse.")
            return pd.DataFrame(columns=["generated_report", "classification"])
        if getattr(batch.request_counts, "failed", 0):
            print(f"Warning: {batch.request_counts.failed} batch requests failed.")

        content = self.client.files.content(batch.output_file_id).text

        reports = {}
        classes = {}
        for line in content.strip().split("\n"):
            if not line:
                continue
            result = json.loads(line)
            output_text = result["response"]["body"]["output"][1]["content"][0]["text"]
            r, c, _ = self._parse_response(output_text)
            reports.update(r)
            classes.update(c)

        report_df = pd.DataFrame({"generated_report": reports, "classification": classes})
        report_df = pd.merge(meta_df, report_df, left_index=True, right_index = True)
        self.save_df(report_df, output_path, out_name)
        return report_df

    def run(self, meta_path: str, output_path, batching: int = 8) -> pd.DataFrame:
        meta_df = pd.read_csv(meta_path, index_col=0)
        stem = Path(meta_path).stem
        jsonl_path = Path(output_path) / (stem + "_batch_input.jsonl")

        self.create_batch_file(meta_df, jsonl_path, batching=batching)
        batch_id = self.submit_batch(jsonl_path)
        batch = self.poll_batch(batch_id)
        return self.retrieve_and_parse(meta_df, batch, output_path, stem + "_reports.csv")
    

    def resume_run(self, batch_id, meta_path: str, output_path) -> pd.DataFrame:
        meta_df = pd.read_csv(meta_path, index_col=0)
        stem = Path(meta_path).stem
        batch = self.poll_batch(batch_id)
        return self.retrieve_and_parse(meta_df, batch, output_path, stem + "_reports.csv")



if __name__ == "__main__":
    generator = LongitudinalReportBatchGenerator()
    generator.run(
        "/path/to/code/BrainDiff/process_mrrate/new_long_samples.csv",
        "/path/to/code/BrainDiff/process_mrrate/data",
    )

    #generator.resume_run("batch_6a5d883a57648190b2ffa5fde0ed5900",
    #                    "/path/to/code/BrainDiff/process_mrrate/data/longitudional_meta.csv",
    #                    "/path/to/code/BrainDiff/process_mrrate/data",
    #)

import json
import re
import csv
import openai
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from create_privacy_span import *

client = openai.OpenAI(api_key="<YOUR_OPENAI_API_KEY>")


SYSTEM_PROMPT = """You are a professional conversation classifier. Analyze the user's query and categorize it into exactly one of the following four categories:
1. "Applications": Questions related to job applications, resumes, CVs, cover letters, or school admission essays.
2. "Financial": Questions related to banking, taxes, investments, billing, financial planning, or insurance.
3. "Emails": Questions related to drafting, editing, or replying to professional or personal emails.
4. "Others": Any queries that do not clearly fall into the three categories above (e.g., general chat, coding, math, creative writing).
Output only the category name: "Applications", "Financial", "Emails", or "Others". Do not include any other text or punctuation.
TEXT:
"""

def get_category(query):
    """2단계: 카테고리 분류 (Others면 탈락시키기 위함)"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"TEXT: {query}"}
            ],
            temperature=0
        )
        category = response.choices[0].message.content.strip().replace('"', '')
        return category
    except:
        return "Others"

# 2. 메인 실행 프로세스
def generate_pupa_dataset():
    print("WildChat 데이터셋 로드 중...")
    # 'streaming=True'를 사용하면 전체를 다운로드하지 않고 실시간으로 읽어옵니다.
    dataset = load_dataset("allenai/WildChat", split="train", streaming=True)

    # dataset = dataset.shuffle(seed=42, buffer_size=10000)

    script_dir = Path(__file__).resolve().parent
    output_file = script_dir / "PUPA_No_PII.csv"
    failure_log_file = script_dir / "PUPA_No_PII_failures.log"
    target_count = 500 # 목표로 하는 클린 데이터 총 개수

    # CSV 헤더 설정
    fields = [
        "conversation_hash", 
        "predicted_category", 
        "user_query", 
        "target_response", 
        "pii_units", 
        "redacted_query"
    ]

    existing_hashes = set()
    collected_count = 0
    if output_file.exists():
        with output_file.open("r", newline="", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                collected_count += 1
                existing_hash = (row.get("conversation_hash") or "").strip()
                if existing_hash:
                    existing_hashes.add(existing_hash)
        print(f"기존 파일 감지: {output_file} (기존 {collected_count}개)")
    else:
        print(f"새 파일 생성: {output_file}")

    if collected_count >= target_count:
        print(
            f"이미 목표 개수({target_count}) 이상입니다. "
            f"현재 누적 개수: {collected_count}"
        )
        return

    write_mode = "a" if output_file.exists() else "w"
    write_header = write_mode == "w"

    with output_file.open(write_mode, newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        if write_header:
            writer.writeheader()

        pbar = tqdm(total=target_count, initial=collected_count, desc="Collecting data")
        failures = 0

        for entry in dataset:
            if collected_count >= target_count:
                break

            if entry.get('language') != 'English':
                continue

            # print(entry)

            conv = entry.get('conversation', [])
            if len(conv) < 2:
                continue
                
            user_query = conv[0]['content']
            target_response = conv[1]['content']

            if not (len(user_query) <= 1500):
                continue

            conv_hash = entry.get('conversation_id', '')
            if conv_hash and conv_hash in existing_hashes:
                continue

            try:
                pii_units, redacted_query = process_user_query_for_no_pii(user_query)
            except Exception as exc:
                failures += 1
                with failure_log_file.open("a", encoding="utf-8") as flog:
                    flog.write(
                        f"conversation_hash={conv_hash}\t"
                        f"error={type(exc).__name__}: {exc}\n"
                    )
                continue

            # 3. PII가 없는(Clean) 데이터만 CSV에 기록
            if pii_units=="":
                # category = get_category(user_query)
                writer.writerow({
                    "conversation_hash": conv_hash,
                    "predicted_category": "", 
                    "user_query": user_query,
                    "target_response": target_response,
                    "pii_units": "", 
                    "redacted_query": redacted_query
                })

                collected_count += 1
                if conv_hash:
                    existing_hashes.add(conv_hash)
                pbar.update(1)
                csvfile.flush()

        pbar.close()

    print(
        f"작업 완료! {output_file}에 총 {collected_count}개의 데이터가 저장되었습니다. "
        f"(이번 실행 실패 {failures}건, 로그: {failure_log_file})"
    )

if __name__ == "__main__":
    generate_pupa_dataset()

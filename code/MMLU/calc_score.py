import csv
import sys
import ast
import os
import pandas as pd


def process_lists(RES_DIR, EXP_NAME, TARGET_CSV):
    # Initialize a dictionary to store the data
    data_dict = {}
    # Loop through all files in the directory
    for filename in os.listdir(f"{RES_DIR}/{EXP_NAME}"):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            with open(os.path.join(RES_DIR, EXP_NAME, filename), 'r') as f:
                # Read the lines
                lines = f.readlines()

                # Check if the file has at least four lines
                if len(lines) >= 4:
                    acc_score = float(lines[0].strip().rsplit(' ', 1)[-1])
                    resp, q_cnt = int(lines[1].strip().split(' ', 1)[0]), len(lines[0].strip().rsplit(' ', 1)[0].split(','))

                    # Add scores to data_dict
                    data_dict[filename[:-4]] = {}
                    data_dict[filename[:-4]]['acc'] = acc_score
                    data_dict[filename[:-4]]['resp'] = resp
                    data_dict[filename[:-4]]['q_cnt'] = q_cnt

    # Write to csv
    with open(TARGET_CSV, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['filename']+ ['acc', 'resp', 'q_cnt'])
        writer.writeheader()
        for filename, scores in data_dict.items():
            row_dict = {'filename': filename}
            row_dict.update(scores)
            writer.writerow(row_dict)


SUBCATEGORY = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

def calculate_score(RES_DIR, EXP_NAME, TARGET_CSV):

    # Process the lists and create a CSV file.
    process_lists(RES_DIR, EXP_NAME, TARGET_CSV)

    # Use pandas to read the CSV file.
    df = pd.read_csv(TARGET_CSV)

    # Calculate the average of the 'acc' column.
    total_q_cnt = df['q_cnt'].sum()
    total_resp_cnt = df['resp'].sum()
    total_correct = sum(df['acc'] * df['q_cnt'])
    # avg_acc = df['acc'].mean()
    avg_acc = total_correct / total_q_cnt
    avg_resp_cnt = total_resp_cnt / total_q_cnt

    print(f"The average accuracy is: {avg_acc}")
    print(f"The average response count is: {avg_resp_cnt}")

    # sub_accs = {}
    # totals = {}
    # for key in CATEGORIES:
    #     sub_accs[key] = []
    #     totals[key] = 0

    # for row in df.itertuples():
    #     sub_cat = row.filename.split('_test')[0] if '_test' in row.filename else row.filename.split('_merge')[0]
    #     sub_cat = SUBCATEGORY[sub_cat][0]
    #     for key in CATEGORIES:
    #         if sub_cat in CATEGORIES[key]:
    #             sub_accs[key].append(row.acc * row.q_cnt)
    #             totals[key] += row.q_cnt
    #             break
    
    # print(sub_accs, totals)
    # for key in CATEGORIES:
    #     print(f"{key}: {sum(sub_accs[key]) / totals[key]}")
    return avg_acc

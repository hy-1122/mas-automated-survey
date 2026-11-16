import csv

with open("artifacts/submission_js.csv", "r", encoding="utf-8") as f1, \
     open("artifacts/test_submission_js.csv", "r", encoding="utf-8") as f2, \
     open("artifacts/full_submission_js.csv", "w", newline="", encoding="utf-8") as fout:

    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    writer = csv.writer(fout)

    header = next(reader1)
    writer.writerow(header)
    writer.writerows(reader1)
    next(reader2)
    writer.writerows(reader2)
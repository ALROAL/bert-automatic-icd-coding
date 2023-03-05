from hydra import initialize, compose
try:
    initialize('./src/')
except ValueError:
    pass
cfg = compose(config_name='config')
import pandas as pd
import re
import csv
from collections import Counter
import operator
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from src.PATHS import RAW_DATA_PATH, PROCESSED_DATA_PATH, SPLIT_DATA_PATH

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(str(code).split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

def write_discharge_summaries(notes_file, out_file):
    #retain only alphanumeric
    tokenizer = RegexpTokenizer(r'\w+')
    print("processing notes file")
    with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            print("writing to %s" % (out_file))
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            #header
            next(notereader)
            i = 0
            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]
                    #tokenize, lowercase and remove numerics
                    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text = '"' + ' '.join(tokens) + '"'
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')
                i += 1
    return out_file

def concat_data(labelsfile, notes_file):
    """
        INPUTS:
            labelsfile: sorted by hadm id, contains one label per line
            notes_file: sorted by hadm id, contains one note per line
    """
    with open(labelsfile, 'r') as lf:
        print("CONCATENATING")
        with open(notes_file, 'r') as notesfile:
            outfilename = '%s/notes_labeled.csv' % PROCESSED_DATA_PATH
            with open(outfilename, 'w') as outfile:
                w = csv.writer(outfile)
                w.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])

                labels_gen = next_labels(lf)
                notes_gen = next_notes(notesfile)

                for i, (subj_id, text, hadm_id) in enumerate(notes_gen):
                    if i % 10000 == 0:
                        print(str(i) + " done")
                    cur_subj, cur_labels, cur_hadm = next(labels_gen)
                    if cur_hadm == hadm_id:
                        w.writerow([subj_id, str(hadm_id), text, ';'.join(cur_labels)])
                    else:
                        print("couldn't find matching hadm_id. data is probably not sorted correctly")
                        break
                    
    return outfilename

def split_data(labeledfile, base_name):
    print("SPLITTING")
    #create and write headers for train, dev, test
    train_name = '%s_train_split.csv' % (base_name)
    dev_name = '%s_dev_split.csv' % (base_name)
    test_name = '%s_test_split.csv' % (base_name)
    train_file = open(train_name, 'w')
    dev_file = open(dev_name, 'w')
    test_file = open(test_name, 'w')
    train_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    dev_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    test_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")

    hadm_ids = {}

    #read in train, dev, test splits
    for splt in ['train', 'dev', 'test']:
        hadm_ids[splt] = set()
        with open('%s/%s_full_hadm_ids.csv' % (SPLIT_DATA_PATH, splt), 'r') as f:
            for line in f:
                hadm_ids[splt].add(line.rstrip())

    with open(labeledfile, 'r') as lf:
        reader = csv.reader(lf)
        next(reader)
        i = 0
        cur_hadm = 0
        for row in reader:
            #filter text, write to file according to train/dev/test split
            if i % 10000 == 0:
                print(str(i) + " read")

            hadm_id = row[1]

            if hadm_id in hadm_ids['train']:
                train_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['dev']:
                dev_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['test']:
                test_file.write(','.join(row) + "\n")

            i += 1

        train_file.close()
        dev_file.close()
        test_file.close()

    return train_name, dev_name, test_name

def next_labels(labelsfile):
    """
        Generator for label sets from the label file
    """
    labels_reader = csv.reader(labelsfile)
    #header
    next(labels_reader)

    first_label_line = next(labels_reader)

    cur_subj = int(first_label_line[0])
    cur_hadm = int(first_label_line[1])
    cur_labels = [first_label_line[2]]

    for row in labels_reader:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        code = row[2]
        #keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_labels, cur_hadm
            cur_labels = [code]
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            #add to the labels and move on
            cur_labels.append(code)
    yield cur_subj, cur_labels, cur_hadm

def next_notes(notesfile):
    """
        Generator for notes from the notes file
        This will also concatenate discharge summaries and their addenda, which have the same subject and hadm id
    """
    nr = csv.reader(notesfile)
    #header
    next(nr)

    first_note = next(nr)

    cur_subj = int(first_note[0])
    cur_hadm = int(first_note[1])
    cur_text = first_note[3]
    
    for row in nr:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        text = row[3]
        #keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_text, cur_hadm
            cur_text = text
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            #concatenate to the discharge summary and move on
            cur_text += " " + text

    yield cur_subj, cur_text, cur_hadm

tokenizer = RegexpTokenizer(r'\w+')
def clean_descriptions(description):
    tokens = [t.lower() for t in tokenizer.tokenize(description) if not t.isnumeric()]
    text = ' '.join(tokens)
    return text

def process_text(text):
    #Re-order
    matches = [x[0] for x in re.finditer("((discharge diagnosis)|(discharge diagnoses)|(past medical history)) (.*?) stopmatch", text)]
    for match in matches:
        re.sub(match, " ", text)
    text = " ".join(matches + [text])
    text = re.sub("stopmatch", "", text)

    return text

if __name__ == "__main__":

    dfproc = pd.read_csv('%s/PROCEDURES_ICD.csv' % RAW_DATA_PATH)
    dfdiag = pd.read_csv('%s/DIAGNOSES_ICD.csv' % RAW_DATA_PATH)

    dfdiag['absolute_code'] = dfdiag.apply(lambda row: str(reformat(str(row[4]), True)), axis=1)
    dfproc['absolute_code'] = dfproc.apply(lambda row: str(reformat(str(row[4]), False)), axis=1)

    dfcodes = pd.concat([dfdiag, dfproc])
    dfcodes.to_csv('%s/ALL_CODES.csv' % PROCESSED_DATA_PATH, index=False,
               columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'absolute_code'],
               header=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
    
    df = pd.read_csv('%s/ALL_CODES.csv' % PROCESSED_DATA_PATH, dtype={"ICD9_CODE": str})
    disch_full_file = write_discharge_summaries(notes_file = '%s/NOTEEVENTS.csv' % RAW_DATA_PATH, out_file="%s/disch_full.csv" % PROCESSED_DATA_PATH)

    print("completed write discharge summaries")
    df = pd.read_csv('%s/disch_full.csv' % PROCESSED_DATA_PATH)
    df = df.sort_values(['SUBJECT_ID', 'HADM_ID'])

    dfl = pd.read_csv('%s/ALL_CODES.csv' % PROCESSED_DATA_PATH)
    dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])

    hadm_ids = set(df['HADM_ID'])
    with open('%s/ALL_CODES.csv' % PROCESSED_DATA_PATH, 'r') as lf:
        with open('%s/ALL_CODES_filtered.csv' % PROCESSED_DATA_PATH, 'w') as of:
            w = csv.writer(of)
            w.writerow(['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ADMITTIME', 'DISCHTIME'])
            r = csv.reader(lf)
            #header
            next(r)
            for i,row in enumerate(r):
                hadm_id = int(row[2])
                #print(hadm_id)
                #break
                if hadm_id in hadm_ids:
                    w.writerow(row[1:3] + [row[-1], '', ''])
    
    dfl = pd.read_csv('%s/ALL_CODES_filtered.csv' % PROCESSED_DATA_PATH, index_col=None)
    dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])
    dfl.to_csv('%s/ALL_CODES_filtered.csv' % PROCESSED_DATA_PATH, index=False)

    sorted_file = '%s/disch_full.csv' % PROCESSED_DATA_PATH
    df.to_csv(sorted_file, index=False)

    labeled = concat_data('%s/ALL_CODES_filtered.csv' % PROCESSED_DATA_PATH, sorted_file)

    #Create train/dev/test splits
    fname = '%s/notes_labeled.csv' % PROCESSED_DATA_PATH
    base_name = "%s/disch" % PROCESSED_DATA_PATH #for output
    tr, dv, te = split_data(fname, base_name=base_name)

    #Filter each split to the top 50 diagnosis/procedure codes
    counts = Counter()
    dfnl = pd.read_csv('%s/notes_labeled.csv' % PROCESSED_DATA_PATH)
    dfnl = dfnl[dfnl["LABELS"].notnull()]
    for row in dfnl.itertuples():
        for label in str(row[4]).split(';'):
            counts[label] += 1

    codes = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    codes_50 = [code[0] for code in codes[:50]]
    codes_100 = [code[0] for code in codes[:100]]
    codes_50_100 = [code[0] for code in codes[50:100]]
    codes_full = [code[0] for code in codes]

    with open('%s/TOP_%s_CODES.csv' % (PROCESSED_DATA_PATH, str(50)), 'w') as of:
        w = csv.writer(of)
        for code in codes_50:
            w.writerow([code])
    
    with open('%s/TOP_%s_CODES.csv' % (PROCESSED_DATA_PATH, str(100)), 'w') as of:
        w = csv.writer(of)
        for code in codes_100:
            w.writerow([code])
    
    with open('%s/TOP_%s_CODES.csv' % (PROCESSED_DATA_PATH, "50_100"), 'w') as of:
        w = csv.writer(of)
        for code in codes_50_100:
            w.writerow([code])

    with open('%s/FULL_CODES.csv' % PROCESSED_DATA_PATH, 'w') as of:
        w = csv.writer(of)
        for code in codes_full:
            w.writerow([code])
    
    for splt in ['train', 'dev', 'test']:
        print(f"{splt}_50")
        hadm_ids = set()
        with open('%s/%s_50_hadm_ids.csv' % (SPLIT_DATA_PATH, splt), 'r') as f:
            for line in f:
                hadm_ids.add(line.rstrip())
        with open('%s/notes_labeled.csv' % PROCESSED_DATA_PATH, 'r') as f:
            with open('%s/%s_50.csv' % (PROCESSED_DATA_PATH, splt), 'w') as of:
                r = csv.reader(f)
                w = csv.writer(of)
                #header
                w.writerow(next(r))
                i = 0
                for row in r:
                    hadm_id = row[1]
                    if hadm_id not in hadm_ids:
                        continue
                    codes = set(str(row[3]).split(';'))
                    filtered_codes = codes.intersection(set(codes_50))
                    if len(filtered_codes) > 0:
                        w.writerow(row[:3] + [';'.join(filtered_codes)])
                    else:
                        w.writerow(row[:3] + [''])
                    i += 1

    for splt in ['train', 'dev', 'test']:
        print(f"{splt}_100")
        hadm_ids = set()
        with open('%s/%s_50_hadm_ids.csv' % (SPLIT_DATA_PATH, splt), 'r') as f:
            for line in f:
                hadm_ids.add(line.rstrip())
        with open('%s/notes_labeled.csv' % PROCESSED_DATA_PATH, 'r') as f:
            with open('%s/%s_100.csv' % (PROCESSED_DATA_PATH, splt), 'w') as of:
                r = csv.reader(f)
                w = csv.writer(of)
                #header
                w.writerow(next(r))
                i = 0
                for row in r:
                    hadm_id = row[1]
                    if hadm_id not in hadm_ids:
                        continue
                    codes = set(str(row[3]).split(';'))
                    filtered_codes = codes.intersection(set(codes_100))
                    if len(filtered_codes) > 0:
                        w.writerow(row[:3] + [';'.join(filtered_codes)])
                    else:
                        w.writerow(row[:3] + [''])
                    i += 1

    for splt in ['train', 'dev', 'test']:
        print(f"{splt}_50_100")
        hadm_ids = set()
        with open('%s/%s_50_hadm_ids.csv' % (SPLIT_DATA_PATH, splt), 'r') as f:
            for line in f:
                hadm_ids.add(line.rstrip())
        with open('%s/notes_labeled.csv' % PROCESSED_DATA_PATH, 'r') as f:
            with open('%s/%s_50_100.csv' % (PROCESSED_DATA_PATH, splt), 'w') as of:
                r = csv.reader(f)
                w = csv.writer(of)
                #header
                w.writerow(next(r))
                i = 0
                for row in r:
                    hadm_id = row[1]
                    if hadm_id not in hadm_ids:
                        continue
                    codes = set(str(row[3]).split(';'))
                    filtered_codes = codes.intersection(set(codes_50_100))
                    if len(filtered_codes) > 0:
                        w.writerow(row[:3] + [';'.join(filtered_codes)])
                    else:
                        w.writerow(row[:3] + [''])
                    i += 1

    for splt in ['train', 'dev', 'test']:
        print(f"{splt}_full")
        hadm_ids = set()
        with open('%s/%s_full_hadm_ids.csv' % (SPLIT_DATA_PATH, splt), 'r') as f:
            for line in f:
                hadm_ids.add(line.rstrip())
        with open('%s/notes_labeled.csv' % PROCESSED_DATA_PATH, 'r') as f:
            with open('%s/%s_full.csv' % (PROCESSED_DATA_PATH, splt), 'w') as of:
                r = csv.reader(f)
                w = csv.writer(of)
                #header
                w.writerow(next(r))
                i = 0
                for row in r:
                    hadm_id = row[1]
                    if hadm_id not in hadm_ids:
                        continue
                    codes = set(str(row[3]).split(';'))
                    filtered_codes = codes.intersection(set(codes_full))
                    if len(filtered_codes) > 0:
                        w.writerow(row[:3] + [';'.join(filtered_codes)])
                    else:
                        w.writerow(row[:3] + [''])
                    i += 1

    #Create descriptions of ICD9 codes
    df_procedures_icd_desc_1 = pd.read_csv('%s/D_ICD_PROCEDURES.csv' % RAW_DATA_PATH, usecols=["ICD9_CODE", "LONG_TITLE"])
    df_procedures_icd_desc_1.columns = ["ICD9_CODE", "TEXT"]
    df_diagnoses_icd_desc_1 = pd.read_csv('%s/D_ICD_DIAGNOSES.csv' % RAW_DATA_PATH, usecols=["ICD9_CODE", "LONG_TITLE"])
    df_diagnoses_icd_desc_1.columns = ["ICD9_CODE", "TEXT"]

    desc_dict = {}
    with open("%s/ICD9_descriptions.txt" % RAW_DATA_PATH, 'r') as labelfile:
        for i,row in enumerate(labelfile):
            row = row.rstrip().split()
            code = row[0]
            if code not in desc_dict.keys():
                desc_dict[code] = ' '.join(row[1:])

    df_icd_desc = pd.DataFrame({"ICD9_CODE":desc_dict.keys(), "TEXT": desc_dict.values()})

    df_procedures_icd_desc_1["ICD9_CODE"] = df_procedures_icd_desc_1["ICD9_CODE"].apply(reformat, is_diag=False)
    df_diagnoses_icd_desc_1["ICD9_CODE"] = df_diagnoses_icd_desc_1["ICD9_CODE"].apply(reformat, is_diag=True)

    df_icd_desc = pd.concat([df_procedures_icd_desc_1, df_diagnoses_icd_desc_1, df_icd_desc])
    df_icd_desc.dropna(subset="ICD9_CODE", inplace=True)
    df_icd_desc.drop_duplicates("ICD9_CODE", inplace=True)

    df_icd_desc["TEXT"] = df_icd_desc["TEXT"].apply(clean_descriptions)

    df_icd_desc_full = df_icd_desc.set_index("ICD9_CODE").loc[codes_full].reset_index()
    df_icd_desc_50 = df_icd_desc.set_index("ICD9_CODE").loc[codes_50].reset_index()
    df_icd_desc_50_100 = df_icd_desc.set_index("ICD9_CODE").loc[codes_50_100].reset_index()
    df_icd_desc_100 = df_icd_desc.set_index("ICD9_CODE").loc[codes_100].reset_index()

    df_icd_desc_full.to_csv('%s/icd_desc_full.csv' % PROCESSED_DATA_PATH)
    df_icd_desc_50.to_csv('%s/icd_desc_50.csv' % PROCESSED_DATA_PATH)
    df_icd_desc_50_100.to_csv('%s/icd_desc_50_100.csv' % PROCESSED_DATA_PATH)
    df_icd_desc_100.to_csv('%s/icd_desc_100.csv' % PROCESSED_DATA_PATH)
    
# coding: utf-8
import os

from reading_comprehension.reading_comprehension_net import ReadingComprehensionModel
from reading_comprehension.data_process import read_examples, convert_examples_to_features

infile = "../input/data.json"
outfile = "../result/result.json"


def main():
    RCModel = ReadingComprehensionModel(model_path=os.path.abspath('./checkpoints/rc_20200709-163056'),
                                        model_selection='B')

    examples = read_examples(full_file=os.path.abspath(infile))
    features = convert_examples_to_features(examples, RCModel.bert_tokenizer, max_seq_length=512, max_query_length=50)

    RCModel.predict(examples=examples, features=features, batch_size=32, test_per_sample=False,
                    support_fact_threshold=0.2, restrict_answer_span=True, result_file=os.path.abspath(outfile))
    print('Done.')


if __name__ == '__main__':
    main()

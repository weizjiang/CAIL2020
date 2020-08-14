# !/usr/bin python
# -*- coding:utf-8 -*-
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from reading_comprehension.reading_comprehension_net import ReadingComprehensionModel
from reading_comprehension.data_process import read_examples, convert_examples_to_features
from reading_comprehension.prepare_data import separate_sentence


if __name__ == '__main__':
    RCModel = ReadingComprehensionModel(model_path=os.path.abspath('../model/checkpoints/rc_20200808-191119'),
                                        model_selection='B')

    passage = "经审理查明:原告刘x0与被告刘x1原6夫妻关系,双方婚姻关系存续期间生育一子刘×3(曾用名:刘×4,2002年9月24日出生)2009年6月5日,刘x0与刘x1经北京市通州区人民法院调解离婚,关于子女抚养,双方约定男孩刘×3由刘x1自行抚养二人离婚后,刘×3随刘x1生活,并在刘x1的原籍读书至小学四年级,后刘x0为刘×3办理转学到本市通州区读书,后刘×3随刘x0生活至今因刘×3已年满十周岁,本院依法征求了刘×3对于变更抚养关系的意见,刘×3称其与父母的关系均很好,现在北京读书与母亲刘x0共同生活,寒暑假与父亲生活,在北京的生活比较习惯,愿意随母亲刘x0生活在本案审理过程中,本院依法向刘x1送达了起诉书及开庭传票后,刘x1未到庭参加诉讼,后向本庭邮寄了一份书面意见,内容为:“本人刘x1同意变更儿子刘×4(刘×3)抚养权问题,同意抚养权变更为刘x0抚养”本庭向刘x0出示了上述书面意见,刘x0表示无异议以上事实,有(2009)通民初字第8185号民事调解书、出生证明、独生子女证、户口登记簿、刘x1书面意见、谈话笔录、开庭笔录等证据在案佐证"
    questions = [
        "刘×3什么时候出生？",
        "刘x1是否参加诉讼？",
        "刘x1为什么没有到法庭参加诉讼？"
    ]

    sentences, _ = separate_sentence(passage)

    context = [sentences[0], sentences]

    qa_data = [
        {
            '_id': str(idx + 1),
            'context': [context],
            'question': question,
            'answer': '',
            'supporting_facts': []
        } for idx, question in enumerate(questions)]

    examples = read_examples(qa_data, max_seq_length=512)
    features = convert_examples_to_features(examples, RCModel.bert_tokenizer, max_seq_length=512,
                                            max_query_length=50)

    # start = timer()

    prediction = RCModel.predict(examples=examples, features=features, batch_size=32, test_per_sample=False,
                                 support_fact_threshold=0.5, restrict_answer_span=True)

    answers = []
    for qa_id, answer in prediction['answer'].items():
        question = qa_data[int(qa_id) - 1]['question']
        answers.append(
            {
                "id": qa_id,
                "question": question,
                "answer": answer if len(question) > 2 else 'unknown',
            }
        )

    print(json.dumps(answers, ensure_ascii=False, indent=4))

    # end = timer()
    # print('- Done in %.3f seconds -' % (end - start))


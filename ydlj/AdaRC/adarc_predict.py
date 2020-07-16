import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../my_libs'))
from adarc.common.predictors.transformer_qa import load_model

# model_path = "./bert_model"
model_path = "./structbert_model"

# GPU: device=0    CPU: device=-1
rc_model = load_model(model_path=model_path, predictor_name="transformer_qa", device=-1)


def predict():
    res = rc_model.predict(question="《战国无双3》是由哪两个公司合作开发的？",
                           passage='''《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，
                                        分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》,
                                        石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。
                                        此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，
                                        请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。
                                        2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。
                                        本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。''')
    print(res['best_span_str'])


def test_cail2020():
    input_file = r'../input/data.json'
    output_file = r'../result/result_adarc.json'

    with open(input_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    answer_dict = {}
    support_fact_dict = {}
    for item in data:
        res = rc_model.predict(question=item['question'],
                               passage=''.join(item['context'][0][1]))
        answer_dict[item['_id']] = res['best_span_str']
        support_fact_dict[item['_id']] = []

    prediction = {'answer': answer_dict, 'sp': support_fact_dict}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(prediction, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':

    test_cail2020()

    # predict()


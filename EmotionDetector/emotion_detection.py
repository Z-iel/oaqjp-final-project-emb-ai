import json
import requests

def emotion_detector(text_to_analyse):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    json_obj = {'raw_document': {'text': text_to_analyse}}
    response = requests.post(url, json = json_obj, headers=headers)
    if response.status_code == 400:
        anger_score = None
        disgust_score = None
        fear_score = None
        joy_score = None
        sadness_score = None
    else:
        response_json = json.loads(response.text)['emotionPredictions'][0]
        anger_score = response_json['emotion']['anger']
        disgust_score = response_json['emotion']['disgust']
        fear_score = response_json['emotion']['fear']
        joy_score = response_json['emotion']['joy']
        sadness_score = response_json['emotion']['sadness']

    output_dict = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
    }
    dominant_emotion = max([(v,k) for k,v in output_dict.items()])
    if dominant_emotion[0] is None:
        output_dict['dominant_emotion'] = None
    else:
        output_dict['dominant_emotion'] = dominant_emotion[1]

    return output_dict

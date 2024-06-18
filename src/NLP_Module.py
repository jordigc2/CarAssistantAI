from transformers import AutoTokenizer, BertForQuestionAnswering, AutoModelForQuestionAnswering, pipeline

import torch

class NLPModule:
    def __init__(self, model_name='deepset/bert-base-uncased-squad2', max_length=200):

        #hugging face token
        self.access_token = "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"

        #Model used for question context identification
        if model_name == 'deepset/bert-base-uncased-squad2':
            self.model = BertForQuestionAnswering.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        #Model for Intent Classification
        self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = ["engine status", "battery information", "general question", "navigation", "music control"]


    def classify_intent(self, user_input):
        result = self.intent_classifier(user_input, self.candidate_labels)
        return result["labels"][0]

    def find_respone(self, question, context):
        
        inputs = self.tokenizer(question, context, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to the appropriate device

        # Generate the output
        outputs = self.model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]

        answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

        return answer
    
    def generate_answer(self, question, context):

        print("----------Generating Answer------------")
        result = self.find_respone(question, context)
        
        return result

if __name__ == "__main__":

    #Unit tests

    nlp_module = NLPModule(model_name='deepset/roberta-base-squad2')

    # questions = ["I want to go to Atomium", "How children should wear the seat belt?", "I want to play 'Mama Mia - ABBA'", "Can you tell me when should I do the engine maintenance?", "Can you tell me how much battery I do have left?"]

    # for question in questions:

    #     result = nlp_module.classify_intent(question)
    #     print("Question: ", question)
    #     print("Intent: ", result)
    #     print("-------")

    question = "How should children be seated?"
    context = "- Context from Manuals: children should always be seated in the rear seats using appropriate child restraint systems the manual emphasizes the importance of wearing seat belts and using child restraint systems. Pregnant women and people suffering from illness should obtain medical advice and wear the seat belt properly"

    result = nlp_module.generate_answer(question=question, context=context)
    print("Question: ", question)
    print("context: ", context)
    print("Answer: ", result)

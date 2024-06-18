from transformers import AutoTokenizer, BertForQuestionAnswering, AutoModelForCausalLM, AutoModelForQuestionAnswering, pipeline
from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain_community.llms import HuggingFacePipeline

import torch

class NLPModule:
    def __init__(self, model_name="TheBloke/wizardLM-7B-HF", max_length=200):

        #hugging face token
        self.access_token = "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"

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

        self.model = LlamaForCausalLM.from_pretrained(model_name,
                                                    # load_in_8bit=True,
                                                    device_map='auto',
                                                    torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True
                                                    )
        self.qa_pipeline = HuggingFacePipeline(pipeline=pipeline("text-generation",
                                            model=self.model,
                                            tokenizer=self.tokenizer,
                                            max_length=1024,
                                            temperature=0,
                                            top_p=0.95,
                                            repetition_penalty=1.15
                                        ))

        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        #Model for Intent Classification
        self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = ["engine status", "battery information", "general question", "navigation", "music control"]


        #Model Used for Casual Text Generation
        # self.text_generator_model = AutoModelForCausalLM.from_pretrained(
        #                                     "openchat/openchat",
        #                                     device_map="auto",
        #                                     token=self.access_token,
        #                                     torch_dtype=torch.bfloat16
        #                                 )
        # self.text_gen_tokenizer = AutoTokenizer.from_pretrained("openchat/openchat")
        # self.text_generator_model.tie_weights()

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
        
        prompt = "Make a cleaner answer from the asked question. Question: " + question + " Answer: " + result
        messages = [
            {"role": "system", "content": "You are a helpful assistant that improves provided answers."},
            {"role": "user", "content": prompt}
        ]

        # text = self.text_gen_tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # model_inputs = self.text_gen_tokenizer([text], return_tensors="pt").to(self.device)

        # generated_ids = self.text_generator_model.generate(
        #     model_inputs.input_ids,
        #     max_new_tokens=512
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(
        #                                         model_inputs.input_ids, 
        #                                         generated_ids
        #                                     )
        # ]

        # response = self.text_gen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return result
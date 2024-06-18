import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from datasets import Dataset, DatasetDict
import evaluate
from accelerate import Accelerator
import numpy as np


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

#https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887

# https://www.reddit.com/r/LocalLLaMA/comments/14vnfh2/my_experience_on_starting_with_fine_tuning_llms/

class CarManualQATrainer:

    def __init__(self, data_path, model_name='t5-small') -> None:
        self.data_path = data_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.lemmatizer = WordNetLemmatizer()
        self.accelerator = Accelerator()
        self.metric = evaluate.load("rouge")
        self.max_length = 50


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def data_loading_cleaning(self):
        df = pd.read_csv(self.data_path)
        df['question'] = df['question'].apply(self.clean_data)
        df['answer'] = df['answer'].apply(self.clean_data)
        self.dataset = df

    def clean_data(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = self.lemmatize_text(text)
        return text

    def lemmatize_text(self, text):
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def prepare_datasets(self):
        self.dataset['input_text'] = self.dataset['question'].apply(lambda x: f"Answer the following question:\nQuestion: {x}\n\nAnswer:")
        self.dataset['target_text'] = self.dataset['answer']

        train_df, val_df = train_test_split(self.dataset, test_size=0.2)
        train_dataset = Dataset.from_pandas(train_df[['input_text', 'target_text']])
        val_dataset = Dataset.from_pandas(val_df[['input_text', 'target_text']])
        dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})

        def tokenize_function(examples):
            model_inputs = self.tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=self.max_length)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(examples['target_text'], truncation=True, padding='max_length', max_length=self.max_length)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['input_text', '__index_level_0__', 'target_text'])

        self.train_dataset = tokenized_datasets['train']
        self.val_dataset = tokenized_datasets['validation']

    def split_data(self, test_size=0.2):
        train, val = train_test_split(self.dataset, test_size=test_size)
        self.train_dataset = train
        self.val_dataset = val

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        print("eval predictions: ", eval_pred)
        if isinstance(predictions, list):
            predictions = torch.tensor(predictions)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        if isinstance(labels, list):
            labels = torch.tensor(labels)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                        for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                        for label in decoded_labels]

        # Debugging: Print some predictions and labels
        print("Sample Predictions: ", decoded_preds[:3])
        print("Sample Labels: ", decoded_labels[:3])

        # Compute ROUGE scores
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)

        # Extract ROUGE f1 scores
        print(result)
        result = {key: value * 100 for key, value in result.items()}

        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id)
                        for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        print("results: ", result)
        return {f'eval_{k}': round(v, 4) for k, v in result.items()}

    def train_model(self, epochs=3, batch_size=16):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        training_args = Seq2SeqTrainingArguments(
            output_dir='./results',
            eval_strategy="steps",
            eval_steps=100,
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="steps",
            save_steps=200,
            learning_rate=4e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs= epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,
            # metric_for_best_model="rouge1",
            report_to="tensorboard",
            logging_dir='./logs'
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator#,
            # compute_metrics=self.compute_metrics
        )

        trainer = self.accelerator.prepare(trainer)

        # if torch.cuda.device_count() > 1:
        #     import torch.distributed as dist
        #     dist.init_process_group(backend='nccl')
        #     model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)
        #     trainer.model = model

        trainer.train()

    def evaluate_model(self):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=Seq2SeqTrainingArguments(
                output_dir='./results',
                per_device_eval_batch_size=16
            ),
            eval_dataset=self.val_dataset#,
            #compute_metrics=self.compute_metrics  # Ensure compute_metrics is used here as well
        )
        results = trainer.evaluate()

        print(results)

    def save_model(self, path='./model'):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path='./model'):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)

    def answer_question(self, question):
        input_text = f"Answer the following question.\nQuestion: {question}\n\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to the appropriate device

        # Generate the output
        outputs = self.model.generate(**inputs,  num_beams=8, do_sample=True, min_length=10, max_length=self.max_length)

        # Decode the generated output
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer

count = 0
if __name__ == "__main__":

    car_manual_qa_trainer = CarManualQATrainer('./data/manuals_qa_pairs.csv', model_name='google/flan-t5-base')
    car_manual_qa_trainer.data_loading_cleaning()
    car_manual_qa_trainer.prepare_datasets()
    car_manual_qa_trainer.train_model(epochs=50)

    print("EVALUATION!!")
    car_manual_qa_trainer.evaluate_model()
    # car_manual_qa_trainer.save_model()

    count += 1
    print(count)

    # Inference
    question = "How to change the car oil?"
    answer = car_manual_qa_trainer.answer_question(question)
    print(f"Question: {question}\nAnswer: {answer}")





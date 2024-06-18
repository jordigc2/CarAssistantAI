import os
from documents_db import DocumentsProcessingDB
from NLP_Module import NLPModule

import streamlit as st

class CarAssistantAI:
    def __init__(self, path_dir, path_db) -> None:
        self.ai_module = NLPModule(model_name='deepset/roberta-base-squad2')

        self.documents_db = DocumentsProcessingDB(path_dir, path_db)
        self.documents_db.start_db()



    def get_answer_ai(self, question, intent):
        context, new_question = self. build_context(question=question, intent=intent)
        if intent == "general question":
            new_question = question

        return self.ai_module.generate_answer(question=new_question, context=context)


    def build_context(self, question, intent):
        context = ""
        new_question = " "
        if intent == "general question":
            context = "\n- Context from Manuals: "
            doc_context = self.documents_db.get_context_question(question)

            for doc in doc_context:
                context += (doc.page_content + ", ")
        elif intent == "music control":
            context = question
            new_question = "Which song I want to play?"
        elif intent == "navigation":
            context = question
            new_question = "Where I want to set the destination to?"

        return context, new_question
    
    def identify_question_intent(self, question):
        intent = self.ai_module.classify_intent(question)

        return intent

def main():

    ai_agent = CarAssistantAI(path_dir="data/manuals/processed", path_db='/workspace/data/vectorized_DB')

    st.set_page_config(page_title="Car AI Agent", page_icon=":robot:")
    st.header("Your Car AI Agent")
    st.write("What can I help up you with?")
    form_input = st.text_input('Enter Query')
    submit = st.button("Generate")

    if submit:
        intent = ai_agent.identify_question_intent(form_input)
        answer = ai_agent.get_answer_ai(form_input, intent=intent)
        if intent == "general question":
            if not len(answer) == 0:
                st.write(answer)
            else:
                st.write("No answer was being found. Try to formulate the question differently.")
        elif intent == "music control":
            st.write("Playing the song  '" + answer + "' in Spotify")
        elif intent == "engine status":
            st.write("Generating SQL code to access engine Status")
        elif intent == "navigation":
            st.write("Setting destination to " + answer + " into the navigator")
        elif intent == "battery information":
            st.write("Generating SQL code to access requested battery information")
        elif intent == "sql generation":
            st.write("Searching request into the DB")

#How can I start using the voice command?

if __name__=="__main__":

    main()

   
    



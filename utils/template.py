"""
To support TRL supervised fine-tuning. Right now, we need to manually set the template here.
"""

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

arabic_alpaca = """أدناه تعليمة تصف مهمة. اكتب استجابة تكميلية مناسبة للطلب.

### التعليمات:
{}

### الاستجابة: {}{}"""

chinese_alpaca = """以下是描述任务的指示。请写出一个适当完成请求的回答。

### 指令:
{}

### 回答: {}{}"""

portuguese_alpaca = """Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{}

### Resposta: {}{}"""

telugu_alpaca = """క్రింది మీకు ఒక టాస్క్‌ను వివరించే నిర్దేశం ఉంది. అభ్యర్థనను సరిగ్గా పూర్తి చేయడానికి సరియైన స్పందన ఇస్తుంది.

### అనుమోదన:
{}

### స్పందన: {}{}"""

russian_alpaca = """Ниже приведена инструкция, описывающая задание. Напишите ответ, который соответственно завершает запрос.

### Инструкция:
{}

### Ответ: {}{}"""

french_alpaca = """Voici une instruction qui décrit une tâche. Écrivez une réponse qui complète correctement la demande.

### Instruction:
{}

### Réponse: {}{}"""

spanish_alpaca = """A continuación se presenta una instrucción que describe una tarea. Escribe una respuesta que complete adecuadamente la solicitud.

### Instrucción:
{} 

### Respuesta: {}{}"""

MedQuad_template = """Below is a conversation between a user and a professional doctor. The doctor gives helpful, precise, detailed and polite answer to the user's question. USER: {} DOCTOR: {}{}"""

vicuna_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}{}"""

multi_turn_template = """{}\n### Response:{}{}"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
    'vicuna': (vicuna_template, ' ASSISTANT:'),
    'medquad': (MedQuad_template, ' DOCTOR:'),
    'arabic_alpaca':(arabic_alpaca,'\n### التعليمات:'),
    'chinese_alpaca':(chinese_alpaca,'\n### 回答：'),
    'portuguese_alpaca':(portuguese_alpaca,'\n### Resposta:'),
    'telugu_alpaca':(telugu_alpaca,'\n### స్పందన:'),
    'russian_alpaca':(russian_alpaca,'\n### Ответ:'),
    'french_alpaca':(french_alpaca,'\n### Réponse:'),
    'spanish_alpaca':(spanish_alpaca,'\n### Respuesta:'),
    'multi-turn':(multi_turn_template,"\n### Response:")
}


def get_formatting_prompts_func(template_name, eos_token,script_args):
    overall_temp, response_temp = TEMPLATE_DICT[template_name]
    def formatting_prompts_func(example):    
        output_texts = []    
        for i in range(len(example['instruction'])):    
            text = overall_temp.format(example['instruction'][i], example['response'][i], eos_token)    
            output_texts.append(text)    
        return output_texts    
    
    return formatting_prompts_func, response_temp

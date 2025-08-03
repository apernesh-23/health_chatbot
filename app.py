from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import base64
import numpy as np
import cv2
from deepface import DeepFace
from io import BytesIO
from PIL import Image
import traceback  

app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_store = {
    "history_ids": None
}

emotion_example_openings = {
    "happy": "It's wonderful to see you looking happy! Would you like to share what's bringing you that joy today?",
    "sad": "I notice you seem a bit down. If you're comfortable, I'm here to listen to what might be on your mind.",
    "angry": "It looks like you might be feeling some strong emotions, perhaps anger. If you'd like to talk about what's going on, I'm here to listen without judgment.",
    "surprise": "That's quite a look of surprise! Did something unexpected happen, or is there something interesting you'd like to share?",
    "fear": "It seems like you might be feeling a bit anxious or uneasy. If you're up for it, could you tell me a little about what's on your mind?",
    "neutral": "You seem to be in a calm space right now. How are things with you, or is there anything you'd like to chat about?",
    "disgust": "I notice a look of displeasure. If something's bothering you or not sitting right, feel free to share what's on your mind."
}


def construct_initial_llm_instruction(emotion):
    """
    Creates the initial instruction for the LLM to start the conversation
    based on the detected emotion. The LLM's output should be the question itself.
    """
    example_opening = emotion_example_openings.get(emotion.lower(
    ), "How are you feeling today? I'm here to listen if you'd like to talk.")

    instruction = (
        f"System: You are a compassionate, empathetic, and understanding mental health support chatbot. "
        f"An image analysis suggests the user might be feeling {emotion}. "
        f"Your primary goal is to gently initiate a supportive conversation. "
        f"Please be the first to speak by asking an open-ended, kind, and inviting question related to their perceived emotion of {emotion}. "
        f"Your question should make them feel comfortable sharing, without being too direct, presumptive, or demanding. "
        f"It should acknowledge the possibility of the emotion softly. "
        f"For instance, if the user seemed sad, a good approach might be something like: \"{emotion_example_openings['sad']}\". "
        f"If they seemed happy, you might ask: \"{emotion_example_openings['happy']}\". "
        f"Given the user seems to be feeling {emotion}, your opening question should be in a similar gentle and open spirit to: \"{example_opening}\". "
        f"It's very important that your entire response is *only* this single opening question to the user. Do not add any other text, explanations, or self-references like 'I will ask...'. Just the question itself.\n"
        f"Chatbot:"  # The model should complete this line with its question.
    )
    return instruction

# def construct_initial_llm_instruction(emotion):
#     """
#     Creates the initial instruction for the LLM to start the conversation
#     based on the detected emotion. The LLM's output should be the question itself.
#     """
#     example_opening = emotion_example_openings.get(
#         emotion.lower(), "How are you feeling today?")

#     # This prompt structure aims to guide DialoGPT to generate the chatbot's part of a dialogue.
#     # "System:" sets the context and task.
#     # "Chatbot:" is a cue for the model to generate what the chatbot would say next.
#     instruction = (
#         f"System: You are an empathetic mental health support chatbot. "
#         f"A user has just been detected and appears to be feeling {emotion}. "
#         f"Your task is to be the first to speak and initiate a supportive conversation "
#         f"by asking a kind and direct question about their {emotion}. "
#         f"For example, if the user seemed sad, you might ask: \"{emotion_example_openings['sad']}\". "
#         f"If they seemed happy, you might ask: \"{emotion_example_openings['happy']}\". "
#         f"Based on the user feeling {emotion}, your actual opening question should be similar to: \"{example_opening}\". "
#         f"Your entire response must be *only* this single opening question to the user, without any other text or explanation.\n"
#         f"Chatbot:"  # The model should complete this line with its question.
#     )
#     return instruction



def get_chatbot_reply(user_input, is_initial_prompt=False):
    """Generate a reply using DialoGPT based on input and existing history"""

    if is_initial_prompt:
        chat_history_store["history_ids"] = None

    if not user_input.strip().endswith(tokenizer.eos_token):
        user_input += tokenizer.eos_token

    new_input_ids = tokenizer.encode(user_input, return_tensors='pt')

    if chat_history_store["history_ids"] is not None:
        current_history_length = chat_history_store["history_ids"].shape[-1]
        max_history_length = 768


        if current_history_length + new_input_ids.shape[-1] > tokenizer.model_max_length - 100:

            history_to_keep = chat_history_store["history_ids"][:, -(
                max_history_length - new_input_ids.shape[-1]):]
            bot_input_ids = torch.cat([history_to_keep, new_input_ids], dim=-1)
        else:
            bot_input_ids = torch.cat(
                [chat_history_store["history_ids"], new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids


    generated_max_length = bot_input_ids.shape[-1] + \
        (50 if is_initial_prompt else 150)

    chat_output_ids = model.generate(
        bot_input_ids,
        max_length=min(generated_max_length, tokenizer.model_max_length),
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75,  
        eos_token_id=tokenizer.eos_token_id)

    reply = tokenizer.decode(
        chat_output_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True  
    ).strip()

    if is_initial_prompt:
        if reply.lower().startswith("chatbot:"):
            reply = reply[len("chatbot:"):].strip()


        prefixes_to_strip = [
            "Okay, I will ask the user:", "Okay, I will ask:", "Okay, I'll ask:",
            "I will ask:", "I'll ask:", "I would say:", "My opening question is:",
            "Sure, I can ask:", "Here's a question:", "How about this:"
        ]
        original_reply_for_stripping = reply  
        for prefix in prefixes_to_strip:
            if reply.lower().startswith(prefix.lower()):
                potential_reply = reply[len(prefix):].strip()
                if potential_reply:  
                    reply = potential_reply
                    break

        if not reply or len(reply.split()) < 3:
            print(
                f"LLM initial reply too short or problematic ('{original_reply_for_stripping}' -> '{reply}'). Falling back.")
            emotion_for_fallback = user_input.split("feeling ")[1].split(
                ".")[0] if "feeling " in user_input else "neutral"
            reply = emotion_example_openings.get(
                emotion_for_fallback.lower(), "How are you feeling today?")


        if reply and not reply[0].isupper() and reply[0].isalpha():
            reply = reply[0].upper() + reply[1:]

        # Remove any trailing incomplete sentences if generation was cut off (less likely with EOS but good practice)
        if '.' in reply and not reply.endswith(('.', '?', '!')):
            reply = reply[:reply.rfind('.')+1]
        elif '?' in reply and not reply.endswith('?'):
            reply = reply[:reply.rfind('?')+1]
        elif '!' in reply and not reply.endswith('!'):
            reply = reply[:reply.rfind('!')+1]


    chat_history_store["history_ids"] = chat_output_ids

    return reply



@app.route('/')
def index():
    chat_history_store["history_ids"] = None 
    return render_template('index.html')


@app.route('/upload-image', methods=['POST'])
def upload_image():
    image_data = request.form.get('image')
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_array_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        emotion, annotated_img_b64 = detect_emotion_and_draw(image_array_cv2)

        initial_instruction_for_llm = construct_initial_llm_instruction(
            emotion)

        print(f"--- Detected emotion: {emotion}")
        print(
            f"--- Initial instruction for LLM (sent to get_chatbot_reply):\n{initial_instruction_for_llm}")

        chatbot_opening_message = get_chatbot_reply(
            initial_instruction_for_llm, is_initial_prompt=True)

        print(
            f"--- Chatbot's opening message (after potential cleaning):\n{chatbot_opening_message}")

        return jsonify({
            "emotion": emotion,
            "reply": chatbot_opening_message,
            "image": f"data:image/jpeg;base64,{annotated_img_b64}" if annotated_img_b64 else None
        })
    except Exception as e:
        print(f"Image processing or LLM error in /upload-image: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to process image or get initial chat reply"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "No message received"}), 400

    try:
        reply = get_chatbot_reply(message)  
        print(f"--- User to Chatbot: {message}")
        print(f"--- Chatbot to User: {reply}")
        
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to get reply"}), 500



def detect_emotion_and_draw(image_array):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    annotated_image_array = image_array.copy()

    try:
        analysis_results = DeepFace.analyze(
            image_array, actions=['emotion'], enforce_detection=False)

        if isinstance(analysis_results, list) and len(analysis_results) > 0:
            analysis = analysis_results[0]
            emotion = analysis['dominant_emotion']
            face_region = analysis['region']

            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            cv2.rectangle(annotated_image_array, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)

            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(annotated_image_array, emotion, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            emotion = "neutral"
            print("DeepFace did not detect a face or returned an unexpected result.")
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(annotated_image_array, (fx, fy),
                              (fx + fw, fy + fh), (255, 0, 0), 2)
            cv2.putText(annotated_image_array, "Face unclear", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', annotated_image_array)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return emotion, encoded_image
    except ValueError as ve: 
        print(f"DeepFace ValueError (likely no face detected): {ve}")
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.1, 4)  
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_image_array, (x, y), (x + w,
                          y + h), (255, 0, 0), 2)  # Blue if OpenCV found
        cv2.putText(annotated_image_array, "Face unclear", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', annotated_image_array)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return "neutral", encoded_image
    except Exception as e:
        print(f"General emotion detection error: {e}")
        traceback.print_exc()
        _, buffer = cv2.imencode('.jpg', image_array)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return "neutral", encoded_image
    
  

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')